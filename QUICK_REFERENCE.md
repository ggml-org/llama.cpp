# Quick Reference: Unresolved Comments Resolution Plan

This document provides a quick reference for addressing the 36 unresolved code review comments. For full details, see `UNRESOLVED_COMMENTS.md`.

## Priority 1: Security Issues (1 comment)

| # | File | Line | Issue | Estimate |
|---|------|------|-------|----------|
| 6 | tools/frameforge/README.md | 140 | Use per-user FIFO location | 2-4 hours |

**Action:** Modify IPC implementation to use `$XDG_RUNTIME_DIR` or per-user `/tmp` directory with proper permissions.

## Priority 2: High Severity (6 comments)

| # | File | Line | Issue | Estimate |
|---|------|------|-------|----------|
| 3 | tools/frameforge/frameforge-ipc.cpp | 162 | Non-blocking I/O error handling | 2-3 hours |
| 4 | tools/frameforge/frameforge-ipc.cpp | 112 | IPCServer callback not invoked | 3-4 hours |
| 11 | tools/frameforge/frameforge-sidecar.cpp | 398 | Server mode not implemented | 4-6 hours |
| 23 | tools/semantic-server/ipc-handler.h | 176 | Windows handle validation | 2-3 hours |
| 24 | tools/semantic-server/ipc-handler.h | 73 | Stop method hang | 2-3 hours |

**Total Estimate:** 13-21 hours

Note: Issue #6 (Security) was previously counted in this section but is now properly categorized under Priority 1 (Security).

## Priority 3: Medium Severity - Critical (12 comments)

### IPC & Communication
| # | File | Issue | Estimate |
|---|------|-------|----------|
| 1 | tools/frameforge/frameforge-ipc.cpp:327 | Message size validation (Windows) | 1 hour |
| 2 | tools/frameforge/frameforge-ipc.cpp:350 | Message size validation (Unix) | 1 hour |
| 25 | tools/semantic-server/ipc-handler.h:223 | Pipe close order | 1 hour |
| 26 | tools/semantic-server/ipc-handler.h:236 | Pipe open failure handling | 2 hours |

### Validation & Error Handling
| # | File | Issue | Estimate |
|---|------|-------|----------|
| 7 | tools/frameforge/frameforge-sidecar.cpp:162 | WAV header alignment | 1 hour |
| 8 | tools/frameforge/frameforge-sidecar.cpp:111 | stoi exception handling | 1 hour |
| 16 | tools/frameforge/frameforge-validator.cpp:101 | Action group validation | 2 hours |
| 19 | tools/frameforge/frameforge-json.cpp:119 | JSON error handling | 2 hours |

### Intent & Sampling
| # | File | Issue | Estimate |
|---|------|-------|----------|
| 9 | tools/frameforge/frameforge-sidecar.cpp:259 | Greedy sampling performance | 3 hours |
| 28 | tools/semantic-server/intent-engine.h:132 | Tokenization failure silent | 1 hour |
| 29 | tools/semantic-server/intent-engine.h:178 | Early stop JSON detection | 2 hours |
| 30 | tools/semantic-server/intent-engine.h:176 | Exception handling too broad | 1 hour |

**Total Estimate:** 18 hours

## Priority 4: Medium Severity - Important (11 comments)

### Testing & Validation
| # | File | Issue | Estimate |
|---|------|-------|----------|
| 5 | tools/frameforge/frameforge-ipc.cpp:362 | Missing IPC test coverage | 4-6 hours |
| 12 | tools/frameforge/frameforge-sidecar.cpp:179 | WAV format validation | 2 hours |
| 13 | tools/frameforge/frameforge-sidecar.cpp:162 | Sample rate validation | 1 hour |
| 17 | tools/frameforge/frameforge-validator.cpp:82 | Missing parameter value tests | 2 hours |
| 18 | tools/frameforge/frameforge-validator.cpp:206 | Additional params test coverage | 2 hours |

### JSON & Parsing
| # | File | Issue | Estimate |
|---|------|-------|----------|
| 20 | tools/frameforge/frameforge-sidecar.cpp:283 | JSON termination check | 2 hours |
| 31 | tools/semantic-server/intent-engine.h:203 | Empty JSON on parse error | 1 hour |
| 32 | tools/semantic-server/intent-engine.h:102 | Lost parse error context | 2 hours |

### Code Quality
| # | File | Issue | Estimate |
|---|------|-------|----------|
| 21 | tools/frameforge/CMakeLists.txt:27 | CMake EXCLUDE_FROM_ALL | 1 hour |
| 33 | tools/semantic-server/semantic-server.cpp:182 | Code duplication | 2 hours |
| 34 | tools/semantic-server/semantic-server.cpp:166 | JSON input validation | 1 hour |

**Total Estimate:** 20-22 hours

## Priority 5: Low Severity (6 comments)

| # | File | Issue | Estimate |
|---|------|-------|----------|
| 10 | tools/frameforge/frameforge-sidecar.cpp:39 | Redundant LLM prompt | 30 min |
| 14 | tools/frameforge/frameforge-sidecar.cpp:153 | WAV buffer size | 15 min |
| 15 | tools/frameforge/frameforge-validator.cpp:61 | Degrees range documentation | 30 min |
| 22 | tools/frameforge/frameforge-sidecar.cpp:227 | Tokenization pattern docs | 30 min |
| 27 | tools/semantic-server/command-validator.h:170 | toupper UB with non-ASCII | 15 min |
| 35 | tools/semantic-server/semantic-server.cpp:169 | is_json_input initialization | 15 min |
| 36 | tools/semantic-server/intent-engine.h:123 | Variable naming | 10 min |

**Total Estimate:** 2.5 hours

## Total Time Estimates

| Priority | Count | Estimated Time |
|----------|-------|----------------|
| Security | 1 | 2-4 hours |
| High | 6 | 13-21 hours |
| Medium (Critical) | 12 | 18 hours |
| Medium (Important) | 11 | 20-22 hours |
| Low | 6 | 2.5 hours |
| **TOTAL** | **36** | **55.5-67.5 hours** |

## Recommended Approach

### Phase 1: Security & Critical Fixes (1-2 weeks)
1. Address security issue (#6)
2. Fix high severity issues (#3, #4, #11, #23, #24)
3. Add basic test coverage

### Phase 2: Stability & Robustness (1-2 weeks)
1. Fix medium severity IPC issues (#1, #2, #25, #26)
2. Improve error handling (#7, #8, #16, #19)
3. Optimize performance (#9)
4. Improve validation (#12, #13, #28, #29, #30)

### Phase 3: Quality & Testing (1 week)
1. Add comprehensive tests (#5, #17, #18)
2. Fix JSON handling issues (#20, #31, #32)
3. Code quality improvements (#21, #33, #34)

### Phase 4: Polish (1-2 days)
1. Fix all low severity issues
2. Documentation updates
3. Final code review

## Quick Start

To get started addressing these issues:

1. **Read full details:** Review `UNRESOLVED_COMMENTS.md` for complete context
2. **Pick a priority group:** Start with Priority 1 (security) or Priority 2 (high severity)
3. **Create a branch:** `git checkout -b fix/issue-N-description`
4. **Make changes:** Follow the suggested fixes in the documentation
5. **Test thoroughly:** Add tests for the fix
6. **Commit with reference:** Include issue number in commit message
7. **Request review:** Create a PR referencing this document

## Testing Strategy

For each fix:
- [ ] Add unit test(s) for the specific issue
- [ ] Verify existing tests still pass
- [ ] Add integration test if applicable
- [ ] Test on multiple platforms (Windows, Linux, macOS)
- [ ] Run with sanitizers (ASAN, UBSAN, TSAN)
- [ ] Profile performance impact if applicable

## Notes

- Many issues can be addressed in parallel by different developers
- Some fixes depend on others (e.g., #4 depends on #11)
- Consider addressing related issues together (e.g., all IPC issues)
- Some estimates are conservative and may be faster with experience

## Owner Requests

The following comments have explicit requests from @TheOriginalBytePlayer to apply changes:

- Comment #1: Message size validation (Windows) - **2 mentions**
- Comment #2: Message size validation (Unix) - **1 mention**
- Comment #9: Greedy sampling performance - **2 mentions**

These should be prioritized.

---

**Last Updated:** 2025-12-29
**Document Version:** 1.0
