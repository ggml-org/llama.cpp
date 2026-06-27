# Deterministic draft: how to run the scenarios and the test suites

Runbook for `external/scripts/run-draft-scenarios.sh` and for the two deterministic-draft
test programs. It exists because several of these steps fail *silently* when run from the
wrong directory or against a stale binary, and a silent skip looks exactly like a pass.

Read this file with:

- `external/README.md` - user facing docs: build, the per-drafter server commands,
  throughput tables, "Verify the Filter" (the same runs this script performs).
- `external/scripts/run-draft-scenarios.sh` - the harness itself (169 lines, self-contained).
- `external/deterministic-draft-filter.md` - the filter/mode design notes.

Everything below assumes the repo is at `/home/samueldoyle/Projects/GitHub/AI/llama.cpp`
and the models are at `/home/samueldoyle/AI_LOCAL/Models`. All paths are relative to the
repo root unless stated otherwise.

- [1. Prerequisites](#1-prerequisites)
- [2. Quickstart](#2-quickstart)
- [3. Running the scenarios](#3-running-the-scenarios)
- [4. What a passing run looks like](#4-what-a-passing-run-looks-like)
- [5. Reading the two result lines](#5-reading-the-two-result-lines)
- [6. Default mode vs intercept-all](#6-default-mode-vs-intercept-all)
- [7. The other test suites](#7-the-other-test-suites)
- [8. Traps that waste the most time](#8-traps-that-waste-the-most-time)
- [9. Error strings and fixes](#9-error-strings-and-fixes)
- [10. Which build produced these numbers](#10-which-build-produced-these-numbers)

## 1. Prerequisites

Three build products must exist and be newer than the sources they come from:

| product | path | build target |
|---|---|---|
| server | `build/bin/llama-server` | `llama-server` |
| regex plugin | `external/plugins/lib/libdeterministic_draft_spec_plugin.so` | `deterministic_draft_spec_plugin` |
| unit test | `build/bin/test-deterministic-draft` | `test-deterministic-draft` |

Configure and build exactly as `external/README.md` "## Build" says. The configuration used
on this machine is `-DGGML_CUDA=ON -DLLAMA_OPENSSL=ON -DDETERMINISTIC_SPEC_ENABLED=ON
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`. `DETERMINISTIC_SPEC_ENABLED=ON`
is what registers the unit test and the plugin target at all.

```sh
cd /home/samueldoyle/Projects/GitHub/AI/llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON -DDETERMINISTIC_SPEC_ENABLED=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j"$(nproc)" --target llama-server test-deterministic-draft
cmake --build build --config Release -j"$(nproc)" --target deterministic_draft_spec_plugin
```

Check staleness before believing any result. A plugin `.so` or server left over from an
older build still runs, and on the same prompt it can reproduce the numbers in
`external/README.md` exactly - which looks like a confirmation and is not one.

```sh
ls -la --time-style=+%Y-%m-%d_%H:%M:%S build/bin/llama-server build/bin/test-deterministic-draft \
          external/plugins/lib/libdeterministic_draft_spec_plugin.so
```

## 2. Quickstart

```sh
# every scenario: 3 default-mode + 3 intercept-all
external/scripts/run-draft-scenarios.sh

# one scenario
external/scripts/run-draft-scenarios.sh default-dspark

# the three default-mode runs only
external/scripts/run-draft-scenarios.sh default-mtp default-dspark default-eagle3
```

Exit status is 0 only if every scenario it ran started, answered and shut down cleanly.

## 3. Running the scenarios

The script is its own runner: it starts `build/bin/llama-server` per scenario, waits for
readiness, sends one completion request, prints the response text and the two result lines,
then kills the server. It keeps its own copy of the scenario table and the request,
mirroring `external/README.md`; the two are kept in sync by hand.

Scenario names, in table order:

```
default-mtp  default-dspark  default-eagle3  ia-mtp  ia-dspark  ia-eagle3
```

`default-*` = filter installed, target still verifies every draft token.
`ia-*` = same plus `--det-draft-intercept-all`, and the rule is forced on (see section 6).

Environment variables:

| variable | default | meaning |
|---|---|---|
| `DET_MODELS` | `/home/samueldoyle/AI_LOCAL/Models` | model root |
| `DET_WORK` | `/tmp/det-draft-scenarios` | where `<name>.log` and `<name>.out` are written |
| `DET_PORT_BASE` | `18080` | first port; ports are handed out one per run |
| `DET_TIMEOUT` | `600` | seconds to wait for the readiness line, per run |

Ports are assigned in run order starting at `DET_PORT_BASE`, so the port a scenario gets
depends on which scenarios you asked for: `run-draft-scenarios.sh` (all six) binds
18080-18085, while `run-draft-scenarios.sh default-dspark` alone binds 18080. The
artifact name, not the port, identifies a run. If a port is already in use the server
exits during startup and the script prints the log tail.

Per run it prints:

```
===== <name> =====
<completion text>
<the grep'd result lines, up to 8>
(log: <DET_WORK>/<name>.log)
```

`<name>.log` holds the full server stdout/stderr, `<name>.out` holds the raw JSON response.

Fixed request, identical for every scenario (`PROMPT` in the script):

```sh
curl -s http://127.0.0.1:$port/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt":"# Write a short story\nOnce upon a time","max_tokens":256}'
```

Fixed server flags appended to every scenario (`FMT` in the script):

```
--reasoning off --temp 0 --top-k 1 --top-p 1.0 -c 8192 -ngl 99 -fa on --jinja
```

The per-scenario arguments are the `-m` target, the `-md` draft, `--spec-type`,
`--spec-draft-n-max 16`, `--det-draft-model <plugin>`, plus `--det-draft-intercept-all`
for `ia-*`. Model pairing is the part that is easy to get wrong:

| scenario | `-m` (target) | `-md` (draft) | `--spec-type` |
|---|---|---|---|
| `*-mtp` | `MTP/Qwen3.5-4B.Q4_K_M.gguf` | none, the target carries its own MTP head | `draft-mtp` |
| `*-dspark` | `dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf` | `dspark/Qwen3-4B-DSpark-Q8_0.gguf` | `draft-dspark` |
| `*-eagle3` | `eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf` | `eagle3/Qwen3-8B-speculator.eagle3-F16.gguf` | `draft-eagle3` |

For DSPARK the file with `Model` in the name is the TARGET and the other is the draft.

Manual single run, if you want to watch it live instead of using the script (default mode):

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
# in another shell:
curl -s http://127.0.0.1:8080/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt":"# Write a short story\nOnce upon a time","max_tokens":256}'
```

Run it from the repo root: `--det-draft-model` is a relative path.

## 4. What a passing run looks like

Measured on this machine (RTX 4070) on commit `b51b617ef`. One run per cell, single request,
`max_tokens 256`, greedy, `--spec-draft-n-max 16`. `accepted/generated` is the
`draft acceptance` line; the rest is the `statistics draft-deterministic` line.

| scenario | acceptance | mean len | #drafts | #truncated | #skipped | pre | post | rejected |
|---|---|---|---|---|---|---|---|---|
| `default-mtp`    | 0.09783 (153/1564) | 2.51 | 101 | 0  | 101 | 1564 | 1564 | 1411 |
| `default-dspark` | 0.34183 (161/471)  | 2.85 | 93  | 42 | 36  | 640  | 471  | 310  |
| `default-eagle3` | 0.08378 (140/1671) | 2.23 | 115 | 12 | 99  | 1797 | 1671 | 1531 |
| `ia-mtp`         | 1.00000 (235/235)  | 14.06 | 20 | 7  | 0   | 311  | 235  | 0    |
| `ia-dspark`      | 1.00000 (217/217)  | 7.03  | 38 | 9  | 0   | 260  | 217  | 0    |
| `ia-eagle3`      | 1.00000 (231/231)  | 13.83 | 24 | 10 | 0   | 372  | 231  | 0    |

Invariants that hold in every cell, and are the fastest way to tell a real run from a
garbled one:

- `#tokens post` equals the `generated` count in the acceptance line (`471` vs `471`).
- `#tokens post = accepted + #target rejected` (`471 = 161 + 310`).
- `ia-*` runs: `#skipped = 0`, `#target rejected = 0`, acceptance exactly `1.00000`.
- `default-*` runs: `#skipped` is large (36-101). That is the plugin declining rounds, not
  an error; see section 6.

Small drift from these numbers is expected between builds, machines and model files. Check
the fingerprint (section 10) before treating a difference as a regression.

Expect this warning in both DSPARK scenarios, in both modes. It is pre-existing and benign:
DSPARK drafts through the dflash implementation, and `--spec-draft-n-max 16` is larger than
the trained block size.

```
common_speculative_impl_draft_dflash: requested draft size (n_max=16, n_min=0) exceeds the trained block size 7 -- clamping to 7
```

## 5. Reading the two result lines

```
slot print_timing: id  0 | task 0 | draft acceptance = 0.34183 (  161 accepted /   471 generated), mean len =  2.85
statistics draft-deterministic: #drafts = 93, #truncated = 42, #skipped = 36, #tokens pre = 640, #tokens post = 471, #target rejected = 310
```

- `draft acceptance` / `mean len` - how many drafted tokens the target verified, and the
  average accepted run length. This line comes from the server's own timing output and is
  not deterministic-draft specific.
- `statistics draft-deterministic` - the filter's own counters:
  - `#drafts` - rounds where the filter saw a draft.
  - `#truncated` - drafts cut short at the first token that failed the rule.
  - `#skipped` - rounds the plugin declined, so the round ran as plain speculative decoding.
  - `#tokens pre` / `#tokens post` - draft tokens before and after the filter.
  - `#target rejected` - draft tokens the target refused.

The counters are cumulative for the process, printed at shutdown, so a log with several
requests shows sums. One request per process keeps a run comparable to the table.

The filter also logs where it came from, once, at startup:

```
common_speculative_init: deterministic draft filter enabled (plugin: external/plugins/lib/libdeterministic_draft_spec_plugin.so, n_max: -1)
```

No such line means the plugin was never installed, and every counter printed after that is
from plain speculative decoding.

## 6. Default mode vs intercept-all

The reference plugin allows letters and whitespace only. In default mode it constrains the
draft head while the target still verifies every surviving token and picks the
bonus/correction token itself. In intercept-all mode (`--det-draft-intercept-all`) the
filter is the sole verifier, so the target veto disappears: acceptance goes to `1.00000`,
`#target rejected` goes to `0`, and output degenerates into repeated-word loops because the
rule, not the model, is choosing.

The plugin also declines a round whose content its rule does not describe, returning
`DETERMINISTIC_DRAFT_REST_SKIP`; that round then runs as plain speculative decoding. The
regex plugin decides this from its own rolling match rate over `DET_DRAFT_REGEX_WINDOW`
(default 256 draft tokens) with `DET_DRAFT_REGEX_MIN_MATCH` (default 0.5) as the threshold,
and it enforces only after a window says the rule applies.

On the prose prompt used here the match rate stays under the default threshold, so
default-mode runs decline nearly every round (`#skipped = 36-101`). That is the fallback
working as designed, and it is why the `ia-*` scenarios in the script force the rule on:

```sh
env DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1 "$SERVER" $args ... --det-draft-intercept-all
```

`external/README.md` "Run All Scenarios" repeats that instruction, which is redundant: the
script applies the two variables itself for `ia-*`. Passing them on the command line again
is harmless. Forgetting them on a manual intercept-all run gives an intercept-all run in
name only, because the plugin declines the rounds.

## 7. The other test suites

There are three separate things that are called "the tests". They are not interchangeable,
and two of them skip in silence when run from the wrong directory.

### 7.1 `ctest` - the whole repo (63 tests)

```sh
ctest --test-dir build -j8 --output-on-failure --timeout 900
```

63 tests (53 main, 6 model, 1 python), about 27 s. The deterministic test is in there
(`test-deterministic-draft`). ctest registers it with `WORKING_DIRECTORY .` = `build/`, and
its two regex-plugin tests locate the plugin relative to the current directory, so under
ctest they print `SKIP (no regex plugin .so found)` and pass. Expected, not a failure.

### 7.2 The registered unit suite - 20 tests

```sh
cd /home/samueldoyle/Projects/GitHub/AI/llama.cpp     # cwd must be the repo root
cmake --build build --target test-deterministic-draft
./build/bin/test-deterministic-draft
```

Source `tests/test-deterministic-draft.cpp`, registered at `tests/CMakeLists.txt` in the
`DETERMINISTIC_SPEC_ENABLED` block. Prints `<name>... OK` per test and
`=== All tests passed ===`. Success looks like 20 `OK` lines and no `SKIP` lines; a `SKIP
(no regex plugin .so found)` means the current directory was wrong. Two of the twenty
(`test_regex_plugin_filter_whitespace_eog`, `test_regex_plugin_skip_fallback`) exercise the
plugin and are the ones that skip; `test_intercept_all_compatible_drafters` parses
`--det-draft-intercept-all` for draft-mtp, draft-dspark, draft-eagle3 and draft-dflash.

### 7.3 The XGrammar POC suite - 34 tests, not in CMake

Source `tests/test-deterministic-draft-xgrammar-poc.cpp`. It has its own `main()`, it is
deliberately NOT registered in `tests/CMakeLists.txt`, and its header says "ARCHIVAL COPY
(NOT BUILT, NOT RUN)". It is still the per-drafter-type suite, and it is the only thing that
exercises `filter_draft` / `apply_bitmask` / `commit_tokens` / `rollback` /
`state_serialization` / bootstrap detection against a real plugin. Build and run it by hand:

```sh
cd /home/samueldoyle/Projects/GitHub/AI/llama.cpp/build/tests
DEFS=$(grep -oP '(?<=^CXX_DEFINES = ).*' CMakeFiles/test-deterministic-draft.dir/flags.make)
INCS=$(grep -oP '(?<=^CXX_INCLUDES = ).*' CMakeFiles/test-deterministic-draft.dir/flags.make)
FLAGS=$(grep -oP '(?<=^CXX_FLAGS = ).*' CMakeFiles/test-deterministic-draft.dir/flags.make)
/usr/bin/c++ $DEFS $INCS $FLAGS -c ../../tests/test-deterministic-draft-xgrammar-poc.cpp -o /tmp/xpoc.o
sed -e 's|CMakeFiles/test-deterministic-draft.dir/test-deterministic-draft.cpp.o|/tmp/xpoc.o|' \
    -e 's|-o ../bin/test-deterministic-draft|-o /tmp/test-xpoc|' \
    CMakeFiles/test-deterministic-draft.dir/link.txt > /tmp/link-xpoc.sh
bash /tmp/link-xpoc.sh
```

Then run it from the plugin's own repo, not from llama.cpp. Its `find_plugin()` looks for
`deterministic-draft.so`, `./deterministic-draft.so`,
`../deterministic-draft-model-poc/build/deterministic-draft.so` and
`./build/deterministic-draft.so`; the XGrammar plugin on this machine lives at
`/home/samueldoyle/Projects/GitHub/AI/deterministic-draft-poc/build/deterministic-draft.so`,
so only `./build/deterministic-draft.so` resolves, and only from that directory:

```sh
cd /home/samueldoyle/Projects/GitHub/AI/deterministic-draft-poc
LLAMA_TEST_MODEL=/home/samueldoyle/AI_LOCAL/Models/MTP/Qwen3.5-4B.Q4_K_M.gguf /tmp/test-xpoc
```

Expected: 34 `OK` lines, `=== All tests passed ===`, exit 0, and startup lines
`--det-draft-intercept-all is enabled for draft-...` plus `[xgrammar-draft] Reusing cached
grammar for 'python'` / `'c'`. From the llama.cpp root instead, 14 of the 34 print
`SKIP (no plugin .so found)` and the suite still exits 0 - which is why running it from the
wrong place is easy to mistake for success. `LLAMA_TEST_MODEL` covers the model-backed tests
(`test_common_checkpoint_roundtrip`, `test_accept_all_sample_and_accept`,
`test_standard_mode_constrains_final_token`).

## 8. Traps that waste the most time

1. **A silent SKIP is not a pass.** Both test programs print `SKIP (...)` and still exit 0.
   Read the output for `SKIP`; never trust the exit status alone.
2. **The unit suite must run from the repo root** and the POC suite from
   `/home/samueldoyle/Projects/GitHub/AI/deterministic-draft-poc`. The plugin paths are
   resolved relative to the current directory in both.
3. **`--det-draft-accept-all` / `det_accept_all` no longer exist.** The flag is
   `--det-draft-intercept-all`, the struct member is `det_intercept_all`, and only the
   accessor keeps the old spelling: `common_speculative_get_det_accept_all`. A file that is
   half renamed does not compile, and the POC suite is not in CMake, so nothing else
   notices.
4. **Stale build products still run.** `build/bin/llama-server` or the plugin `.so` from an
   older commit will happily produce numbers that look like `external/README.md`. Check the
   mtimes, and check the fingerprint in the response JSON.
5. **The `ia-*` runs are only intercept-all because the script forces the rule on.** Any
   manual intercept-all run without `DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1`
   measures the decline fallback and looks broken.
6. **Ports are assigned in run order, not per scenario name.** Asking for one scenario
   moves it to `DET_PORT_BASE`. A "port already in use" failure comes from a server left
   running by an earlier manual run.
7. **`--det-draft-model` needs a compatible `--spec-type`.** Without one of draft-mtp,
   draft-eagle3, draft-dspark, draft-dflash the server exits at startup rather than running
   unfiltered. That is the documented hard-fail case in `external/README.md`.
8. **One request per process** keeps the counters comparable with the tables. The
   `statistics draft-deterministic` line is cumulative and printed at shutdown.
9. **The script's scenario table and `external/README.md` are two copies of the same
   examples.** Changing one does not change the other.
10. **The DSPARK `clamping to 7` warning is normal** in both modes; do not chase it.

## 9. Error strings and fixes

| string | cause | fix |
|---|---|---|
| `--det-draft-model requires a compatible drafter type (--spec-type draft-mtp, draft-eagle3, draft-dspark, or draft-dflash)` | `--det-draft-model` given without a compatible `--spec-type`; server exits 1 | add the matching `--spec-type`, or drop `--det-draft-model` |
| `'struct common_params_deterministic_draft' has no member named 'det_accept_all'; did you mean 'det_intercept_all'?` | file still uses the pre-rename member name | rename the member (`det_accept_all` -> `det_intercept_all`) and the flag (`--det-draft-accept-all` -> `--det-draft-intercept-all`); leave `common_speculative_get_det_accept_all` alone |
| `SKIP (no regex plugin .so found)` (unit suite) | run from somewhere other than the repo root | `cd` to the repo root and rerun |
| `SKIP (no plugin .so found)` (POC suite) | run from somewhere other than the plugin's repo | `cd /home/samueldoyle/Projects/GitHub/AI/deterministic-draft-poc` and rerun, with `LLAMA_TEST_MODEL` set |
| `load: failed to load plugin '<path>'` | wrong `--det-draft-model` path, or plugin not built | build the `deterministic_draft_spec_plugin` target; pass the path relative to the repo root |
| `error: build/bin/llama-server not built (see README.md Build)` | nothing built yet | build the three products in section 1 |
| `error: external/plugins/lib/libdeterministic_draft_spec_plugin.so not built` | plugin missing | `cmake --build build --target deterministic_draft_spec_plugin` |
| `server exited during startup:` + log tail | bad flags, missing model, or the port is taken | read the tail; check `--port`, the model paths, and `--spec-type` |
| `timed out waiting for server` | model load slower than `DET_TIMEOUT` | raise `DET_TIMEOUT`; confirm the readiness line is `llama_server: listening on` |
| `common_speculative_init: deterministic draft filter enabled (plugin: ..., n_max: -1)` | not an error | the filter loaded; `n_max: -1` means the plugin's own cap governs |
| `requested draft size (n_max=16, n_min=0) exceeds the trained block size 7 -- clamping to 7` | DSPARK drafts through the dflash impl, block size 7 | ignore, both modes, pre-existing |
| `synthetic speculative acceptance is enabled for benchmarking; generated output is not valid` | a synthetic-acceptance knob got set | unset it; the scenarios do not use it |

## 10. Which build produced these numbers

The response JSON carries the build and commit that produced it:

```sh
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["system_fingerprint"])' /tmp/det-draft-scenarios/ia-dspark.out
# b11816-b51b617ef
```

`b11816` is the build number, `b51b617ef` the commit. Two different fingerprints mean two
different binaries, so never compare their counters as if they were one series.

The tables in `external/README.md` were recorded on `b11383-d450a63fb`, before the branch was
rebased onto current upstream. The answers on the current commit are shaped the same, with
the same decline behaviour and the same invariants, but the raw counters differ by a few
percent. If you need to reproduce the README numbers exactly, check out the commit named in
that section, rebuild, and re-run; do not try to make the current tree match them.

Related local notes from earlier sessions: a pre-rebase run log is at
`/tmp/opencode/detrun/` and a post-rebase one at `/tmp/det-draft-scenarios/`; they are
scratch files, not part of the repo.
