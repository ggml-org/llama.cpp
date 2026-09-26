# PR #24 gh-resolve triage report

PR: https://github.com/Raudbjorn/ggml-llama.cpp/pull/24
Head: ffca3ffd3 (`fix(sycl): thread tile_route and CLI render_node through helpers`)
Base: master @ 950a171cd
Outcome: 9 unresolved threads at start, 0 at end. `gh-resolve list 24` -> "No unresolved threads found".

## Real bugs (fixed)

| Thread ID | Source | File | Bug | Fix |
|---|---|---|---|---|
| PRRT_kwDOTGTyNs6TaY0Y | sourcery-ai | scripts/perf/bench_submission.py:289 | `prepare_state` hardcoded `/dev/dri/renderD128` for `check_sole_tenancy` while `run_sample` honored `--render-node`. | Added `render_node: str` to `prepare_state(...)` and threaded `args.render_node` from `main()`. |
| PRRT_kwDOTGTyNs6Taaf0 | copilot-pull-request-reviewer | ggml/src/ggml-sycl/fattn-common.hpp:1385-1387 | `ggml_sycl_fattn_profile_record(...)` was called with `need_f16_K || need_f16_V` as its first argument, but the C++ side defines that first parameter as `bool tile_route`. The bucketing in `fattn.cpp:241-242` would misclassify every q8_0 launch. | Added `bool tile_route` to `launch_fattn(...)`. The 6 TILE call sites in fattn-tile.hpp now pass `true`; the VEC call site in fattn-vec.hpp passes `false`. The profile record call now passes the actual route flag. |
| PRRT_kwDOTGTyNs6Tq7MG | copilot-pull-request-reviewer | scripts/test_bench_spec.py:13-15 | Module-load `assert SPEC is not None and SPEC.loader is not None` is silently stripped under `python3 -O`, defeating the contract. | Replaced with an explicit `if SPEC is None or SPEC.loader is None: raise RuntimeError(...)`. Verified under `python3 -O -m unittest scripts.test_bench_spec` -> 6 tests pass. |

## False alarms (verified and dismissed)

| Thread ID | Source | File | Reviewer's claim | Why it is a false alarm |
|---|---|---|---|---|
| PRRT_kwDOTGTyNs6TaY0a | sourcery-ai | scripts/perf/bench_submission.py:258-264 | `subprocess.Popen` without a static string is command-injection-risky. | The `command` list is built by `server_command(...)` from argparse-validated absolute paths (`server_bin`, `model`) and static CLI tokens; the only interpolated value is `args.port` (int). No shell call, no user-controlled argv. The opengrep `dangerous-subprocess-use-audit` rule is a known noisy false positive on arg-parsed subprocess invocations. |
| PRRT_kwDOTGTyNs6TaY0b | sourcery-ai | scripts/perf/server_state_roundtrip.py:183-189 | Same as above. | Same reason: `command` is built from `args.server_bin`, `args.model`, `args.port`, `out_dir` (all argparse-resolved) and static CLI tokens. `grep -c 'shell=True' scripts/perf/server_state_roundtrip.py` returns 0. |
| PRRT_kwDOTGTyNs6TabK7 | gemini-code-assist | ggml/src/ggml-sycl/sycl-mutable-command-list-probe.cpp:206-214 | Calling `zeCommandListClose` after `zeCommandListUpdateMutableCommandsExp` is undefined behavior because the update already closes the list. | Authoritative spec at /usr/include/level_zero/ze_api.h:15855-15856 (system header for the Level Zero runtime installed on this host) states: "The application must close a mutable command list after completing all updates." The function does NOT close the list; the application is responsible for closing it. The probe's `appendLaunchKernel -> close -> execute -> updateMutableCommands -> close -> execute` pattern is the canonical Level Zero mutable-command pattern; removing the second close would skip the second execute path. |
| PRRT_kwDOTGTyNs6TabK- | gemini-code-assist | scripts/perf/bench_spec.py:205-211 | `row.get("id")` on line 211 will raise AttributeError if `row` is not a dict. | The line above already guards the path: `logprob = row.get("logprob") if isinstance(row, dict) else None`. The `elif row.get("id") != token` line on 211 is reachable only when `isinstance(row, dict)` was True, so no AttributeError is possible. The reviewer's proposed unconditional `.get(...)` matches the existing pattern. |
| PRRT_kwDOTGTyNs6TabLA | gemini-code-assist | scripts/perf/server_state_roundtrip.py:84-99 | A global `request_json` exists at line 104 and `make_requester` should be reduced to a lambda that delegates to it to remove duplication. | `grep -n 'def request_json' scripts/perf/server_state_roundtrip.py` returns only line 85 -- the closure inside `make_requester`. There is no global duplicate; the proposed refactor would just rename the existing closure and lose the explicit `RequestJson` return type. |
| PRRT_kwDOTGTyNs6Tq7L9 | copilot-pull-request-reviewer | scripts/test_server_state_roundtrip.py:27-29 | A blank line between `@unittest.skipUnless(...)` and the class definition is a SyntaxError. | A blank line between a decorator and a class definition is valid Python and PEP 8 compliant. `python3 -c 'import ast; ast.parse(open("scripts/test_server_state_roundtrip.py").read())'` parses without error. The reviewer is wrong; the file is well-formed and the full 78-test campaign suite runs it cleanly. |

## Verification

- `python3 -m unittest scripts.test_bench_spec scripts.test_bench_a770_fork_unique scripts.test_server_state_roundtrip scripts.test_sweep_a770_mmvq_geometry scripts.test_bench_sycl_cold_jit` -> 78 tests pass.
- `python3 -O -m unittest scripts.test_bench_spec` -> 6 tests pass under -O.
- `git diff --check` -> clean.
- `git diff --staged` (pre-commit) -> only the targeted render_node, tile_route, and assert fixes.
- `icpx -fsyntax-only` on `template-instances/fattn-tile-instance-dkq128-dv128.cpp` and `template-instances/fattn-vec-instance-f16-q8_0.cpp` with the project's full build flags -> rc=0 for both.
- `git log origin/p5-post-performance-campaign` -> head ffca3ffd3 now contains all three fixes.
- `gh-resolve list 24 -r Raudbjorn/ggml-llama.cpp` -> "No unresolved threads found."
