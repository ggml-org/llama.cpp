# PR #31 Review-Thread Resolution Report

## PR
- **PR**: #31 - `p324-quality-gate-fixes` -> `master`
- **Title**: `fix(quality-gate): harden correctness/PPL/scaling checks against stale review`
- **State**: Open, mergeable (CLEAN)
- **File touched**: `scripts/turbo-quality-gate.sh`

## Branch State
- **Branch**: `p324-quality-gate-fixes`
- **Merge commit**: `94b2458f9` (merge origin/master)
- **Additional fixes**: `0a9b2c7fd` (collapse STRICT guard, portable mktemp)

## Methodology
1. Fetched master into `p324-quality-gate-fixes` (PR #31's branch).
2. Merged `origin/master` -> conflict in `scripts/turbo-quality-gate.sh`.
3. Resolved conflicts by taking master's version (which supersedes the
   `is_strict_clean_run` helper with numeric xfail/skip/xpass parsing).
4. Fixed a pre-existing syntax error in master (`fi` -> `}` after the
   `|| { ... }` brace group in the first-pass GATE-FAIL guard).
5. Collapsed a redundant nested `if [ "$STRICT" = "1" ]` guard in the
   turbo-FA second pass (thread 3).
6. Replaced the non-portable mktemp fallback with a portable form
   (thread 5). Verified with `TMPDIR="/tmp/has spaces/"` + a fake mktemp
   that rejects the bare `mktemp -d` invocation.
7. Committed as `94b2458f9` (merge) + `0a9b2c7fd` (additional fixes).
8. Pushed to origin.
9. Listed unresolved threads via `gh-resolve list 31`.
10. Posted substantive per-thread replies citing file:line + commit SHA.
11. Resolved all 6 threads via `gh-resolve resolve <thread_id>`.

## Verdict Summary

| # | Thread ID | Reporter | Verdict | Resolution |
|---|-----------|----------|---------|------------|
| 1 | PRRT_kwDOTGTyNs6Tt-WI | gemini-code-assist | Real bug | Fixed by master merge |
| 2 | PRRT_kwDOTGTyNs6Tt-s7 | copilot | Real bug | Fixed by master merge |
| 3 | PRRT_kwDOTGTyNs6Tt-tL | copilot | Real bug | Fixed by 0a9b2c7fd |
| 4 | PRRT_kwDOTGTyNs6Tt-tQ | copilot | Real bug | Fixed by master merge |
| 5 | PRRT_kwDOTGTyNs6Tt-tZ | copilot | Real bug | Fixed by 0a9b2c7fd |
| 6 | PRRT_kwDOTGTyNs6TuCWN | copilot | Real bug | Fixed by merge + additional commits |

**Total real bugs: 6 | False alarms: 0**

## Detailed Thread Analysis

### Thread 1 (PRRT_kwDOTGTyNs6Tt-WI) - gemini-code-assist [critical]
- **Claim**: `is_strict_clean_run` has two critical bugs:
  1. Regex `0 XPASS,` does not match the actual harness format
     `0 XPASS (promote to GATE!)`.
  2. Negated `[0-9]+ xfail` regex rejects clean runs because `[0-9]+` matches `0`.
- **Verdict**: REAL BUG. Both points are correct. The function as written
  would always fail in strict mode.
- **Resolution**: The `is_strict_clean_run` helper was removed entirely.
  Master's numeric parsing approach (extracted via `grep -ioE '[0-9]+ xfail'`
  then checked with `[ "${xfail_n:-0}" -gt 0 ]`) correctly handles both
  the format mismatch and the `0` case.

### Thread 2 (PRRT_kwDOTGTyNs6Tt-s7) - copilot
- **Claim**: Same root cause as Thread 1.
- **Verdict**: REAL BUG. Correct identification of the same flaw.
- **Resolution**: Fixed by the same mechanism as Thread 1.

### Thread 3 (PRRT_kwDOTGTyNs6Tt-tL) - copilot
- **Claim**: Redundant nested `if [ "$STRICT" = "1" ]` in the turbo-FA
  second pass (previously lines 165-174). The else branch is unreachable.
- **Verdict**: REAL STYLE BUG. The outer guard at line 151 already ensures
  strict mode; the inner guard and its else were dead code.
- **Resolution**: Collapsed to a single flat block in commit 0a9b2c7fd.

### Thread 4 (PRRT_kwDOTGTyNs6Tt-tQ) - copilot
- **Claim**: `validate_numeric` comment said "positive" but regex accepts `0`.
- **Verdict**: REAL BUG (at the time of review). The comment was inaccurate.
- **Resolution**: Master's commit 7be8ce58c corrected the comment to
  "returns 0 if numeric, 1 if missing/non-numeric" (line 188 in current
  HEAD), which accurately describes the non-negative check.

### Thread 5 (PRRT_kwDOTGTyNs6Tt-tZ) - copilot
- **Claim**: `mktemp -d -t turbo-gate.XXXXXX` is not portable between
  macOS and Linux.
- **Verdict**: REAL BUG. The `-t` flag has different semantics on BSD mktemp
  vs GNU mktemp.
- **Resolution**: Replaced with `mktemp -d "${TMPDIR:-/tmp}/turbo-gate.XXXXXX"`
  in commit 0a9b2c7fd. Verified by creating a fake mktemp that rejects the
  bare `mktemp -d` invocation, setting `TMPDIR="/tmp/has spaces/"`, and
  confirming the fallback creates the directory correctly.

### Thread 6 (PRRT_kwDOTGTyNs6TuCWN) - copilot
- **Claim**: PR description lists several functional changes
  (mktemp portability, strict-mode helper, early-return, -x checks,
  METRIC_VALID removal) but the diff only fixes a syntax error.
- **Verdict**: REAL BUG (at the time of review). The description was ahead
  of the diff.
- **Resolution**: Not a false alarm - the mismatch was real at review time.
  After merging master (94b2458f9) and applying additional fixes
  (0a9b2c7fd), all five described changes are now present in the file:
  1. mktemp portability at line 44
  2. Strict-mode numeric parsing at lines 127-131, 163-167
  3. Early-return at lines 119-136
  4. -x checks at the top of `stage_ppl` and `stage_scaling`
  5. METRIC_VALID removal in `validate_numeric` (lines 188-196)

## Smoke-Test Evidence

```
# Normal smoke test (missing binaries)
$ LLAMA=/nonexistent CORRECTNESS_BIN=/nonexistent bash scripts/turbo-quality-gate.sh
FAIL | 0.1 correctness (LLAMA_TEST_TURBO_FA=0) (reason=binary missing at /nonexistent)
SKIP | 1 perplexity (turbo3 vs q8_0, -fa on) (MODEL unset (non-strict))
SKIP | 2 context-scaling ratio (MODEL or WIKI unset (non-strict))

# Portability test (thread 5 verification)
$ TMPDIR="/tmp/has spaces/" PATH="$FAKE_BIN:$PATH" bash -c '...'
STAGE_LOG_DIR=[/tmp/has spaces//turbo-gate.NLNdYp]
OK: directory created via fallback

# Parse check
$ bash -n scripts/turbo-quality-gate.sh
PARSE OK
```

## Commands Used

```bash
# Fetch and merge master
git fetch origin master
git merge origin/master

# Resolve conflicts (take master's version)
git checkout --theirs scripts/turbo-quality-gate.sh

# Apply additional fixes (threads 3 and 5)
# (edit tool: collapse nested STRICT guard, fix mktemp fallback)

# Verify
bash -n scripts/turbo-quality-gate.sh

# Commit
git commit -am "fix(quality-gate): collapse redundant nested STRICT guard, portable mktemp fallback"

# Push
git push origin p324-quality-gate-fixes

# Reply + resolve
python3 /tmp/reply_resolve_p31.py

# Verify
gh-resolve list 31  # no unresolved threads
```

## Follow-up Actions
- PR #31 is mergeable (CLEAN). The user can merge it into master directly.
- All 8 review threads on the previous PR (#28) were already resolved in
  the prior session; PR #31's threads are now also fully resolved.
