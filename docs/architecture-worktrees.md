# Worktree Layout

Each hardening change lands on its own feature branch in an isolated
git worktree. The orchestrator (the user's representative in the
conversation) merges the branches into `main` after the per-branch
tests pass.

## Setup

The main checkout is at:

```
/Users/user/Developer/GitHub/tessera
```

with `main` tracking `tribunus-dev/tessera`.

Each agent's worktree is at:

```
/Users/user/Developer/GitHub/tessera.worktrees/<branch-name>
```

with its own working directory, but sharing the same `.git` as the
main checkout.

## Branches

| Branch | Agent scope | Merge target |
|--------|-------------|--------------|
| `tessera/arg-cpp-dedup` | Fix duplicate CLI flag registrations; add docs for `--telemetry-*` and `--no-embedded-mtp` | `main` |
| `tessera/dft-observer` | Replace `dft.` string prefix in `llama-graph.cpp` with per-context observer state | `main` |
| `tessera/dflash-gemma4` | Extract gemma4-specific extras from `dflash.cpp` into a `llama_model_dflash_gemma4` subclass | `main` |
| `tessera/spec-calib-api` | Extract manual drafter forward from `imatrix.cpp` into `common/speculative-calibration.{h,cpp}` | `main` |
| `tessera/auto-mtp-fix` | Decide and implement auto-MTP fix (implement mtp_context() or remove auto-trigger) | `main` |
| `tessera/telemetry-schemas` | Unify v1/v2 telemetry schemas; deprecate v1 | `main` |
| `tessera/tests` | Add production-grade tests for dflash loader, dspark markov head, telemetry, server-context | `main` |

## Creating a worktree

For each agent, the orchestrator creates the worktree:

```sh
cd /Users/user/Developer/GitHub/tessera
git worktree add -b tessera/<branch-name> /Users/user/Developer/GitHub/tessera.worktrees/<branch-name> main
```

The agent then works in the worktree directory.

## Merging

When the agent's branch is ready:

```sh
cd /Users/user/Developer/GitHub/tessera
git fetch juliantorr-llama <branch-name> 2>/dev/null || git fetch origin <branch-name>
git checkout main
git merge --no-ff tessera/<branch-name>
# run full test suite
# if tests pass:
git push origin main
```

If the merge has conflicts, the orchestrator resolves them by hand
or by spawning a conflict-resolution agent.

## Cleanup

After the merge, the worktree is removed:

```sh
git worktree remove /Users/user/Developer/GitHub/tessera.worktrees/<branch-name>
git branch -d tessera/<branch-name>
```
