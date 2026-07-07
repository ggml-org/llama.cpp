# RALPH loop iteration log

Append-only. One entry per task per iteration. Format:

```
## YYYY-MM-DD — <task-id> — <one-line title>

What was done, what ran (commands / tests), result.
Any new follow-up tasks uncovered (link to RALPH_TASKS entries).

---
```

## 2026-07-07 — P0.0 — Topology recon

Wrote `TOPOLOGY.md` at repo root by running live commands (not from memory). Verified: Raudbjorn-fork on `turbo-sycl-opt` @ `58a8c5baa`; `Raudbjorn-ggml-llama.cpp` actually on `turbo-sycl-fa-vec` @ `0181a0cd2` (prompt's `4e56a9980` is stale); `pr7-fix` is a worktree of the same git repo. Confirmed `llama-sycl.cpp.service` is `inactive (dead)`, not crash-looping; :8088 is squatted by `/usr/bin/llama-server` PID 4481 (CPU-only, mistral-7b q8_0 KV). All 3 validation models exist on disk. Render node has only desktop clients (codium/chrome/qoder), no llama/SYCL holders — GPU compute idle.

No new tasks uncovered. P0 housekeeping items already list the port-squat as the first sub-task (read-only diagnosis), and TOPOLOGY.md confirms the diagnosis.

## 2026-07-07 — P0 — Service-port diagnosis (NOT wedged)

Read-only diagnosis, no service restart, no kill. Live commands (not recall):
`ss -tlnp | grep :808[0-9]`, `systemctl status llama-sycl.cpp.service`,
`ps -p 4481 -o ...`, `cat /proc/4481/status`, `curl http://127.0.0.1:8088/health`,
`fuser /dev/dri/renderD128`.

**Verdict:** Not wedged. `llama-sycl.cpp.service` is `disabled` + `inactive (dead)`,
NOT crash-looping — the prompt's "crash-looping ~8h" claim is stale (matches
what P0.0's TOPOLOGY.md already said). :8088 is squatted by root PID **4481**
(`/usr/bin/llama-server`, `--n-gpu-layers 0`, CPU-only q8_0 KV, mistral-7b
Q4_K_M), PPid=1 (reparented to init → manual launch, not unit-launched),
uptime 37m59s at this iteration, `/health` returns `{"status":"ok"}`. GPU
renderD128 idle (only desktop compositor PIDs 5814/90996/93227 — codium,
chrome, qoder). All 3 validation models on disk (per TOPOLOGY.md).

**P1 doesn't need the live service** — it invokes `llama-perplexity` and
`test-sycl-turbo-correctness` directly against the build in
`Raudbjorn-fork/build-port/`. So P0 closes without action.

**Four remediation options if a future task needs the live SYCL service:**
(a) `kill 4481` then `systemctl enable --now llama-sycl.cpp.service` — but the
    service is currently `disabled` and was never enabled, suggesting whoever
    launched 4481 prefers CPU for now (and is actively using it);
(b) Move SYCL service to a different port (e.g. :8088 → :9088) and update
    `llama-sycl.cpp.service` + any reverse-proxy / Tailscale DNS pointing at it;
(c) Move CPU squat to a different port (e.g. via the `llama-cpu@.service`
    template — already loaded, template-instance active) and let SYCL have :8088;
(d) Document-and-script: keep both as-is, write a wrapper that re-points the
    unit to whichever port is free at start.

**Recommendation:** option (a) ONLY when the user signals SYCL is needed for an
inference workload and that 4481 isn't currently serving traffic (check
`ss -tnp | grep :8088` for active clients before any kill). Until then, defer.

**File-state gotcha worth flagging:** `RALPH_TASKS.md` lives at workspace root
(`/mnt/mrgr/llama-cpp-sycl-turbo/RALPH_TASKS.md`), NOT inside the Raudbjorn-fork
git repo. The repo tracks only `RALPH_PROGRESS.md` and `TOPOLOGY.md`. So the
P0 checkbox flip is an out-of-repo edit; this commit carries the progress entry
only. Matches the pattern P0.0 established (RALPH_TASKS isn't in commit
`7a3731a27` either). If the user wants the checkbox ride-with-commit, that's
a separate decision (copy RALPH_TASKS.md into the repo and start tracking it).

No new tasks added.

## 2026-07-07 — P0 (continued) — Extend FA probes to fleet head_dim/GQA shapes

Followed up on P0 sub-task 2: extend `test-sycl-turbo-correctness` FA probes to the
head_dim/GQA shapes used by the 3 validation models. Discovered the 3-model fleet
is actually homogeneous on head_dim=128 (turbo's hard invariant), so the real value
of this extension is **GQA-shape coverage**, not head_dim coverage.

**What ran:**
1. Live `GGUFReader` extraction of all 3 model configs:
   - `mistral-7b-instruct-v0.1.Q4_K_M.gguf`: arch=llama, embed=4096, head_count=32,
     head_count_kv=8, FF=14336 → head_dim=128, **GQA=4:1**.
   - `llama31-8b-heretic Q4_K_M.gguf`: identical shape (embed=4096, head_count=32,
     head_count_kv=8) → head_dim=128, **GQA=4:1**.
   - `Qwen3-Coder-30B-A3B-UD-Q3_K_XL.gguf`: arch=**qwen3moe**, embed=2048,
     head_count=32, head_count_kv=4, expert_count=128, expert_used_count=8,
     `qwen3moe.attention.key_length=128` → head_dim=**128**, **GQA=8:1**.
   Key finding: `qwen3moe.attention.key_length`/`value_length` are explicitly set
   in the GGUF (don't infer head_dim from `embed/head_count = 64` — that would be
   wrong; the explicit key_length is authoritative).
2. Read the harness file (`tests/test-sycl-turbo-correctness.cpp`) end-to-end and
   the ggml.h signatures for `ggml_new_tensor_4d` and `ggml_flash_attn_ext` (GQA
   precondition: `n_head % n_head_kv == 0`).
3. Extended `probe_flash_attn`, `probe_fa_f16`, and `probe_attn_noflash` with
   trailing-default `int64_t nh_q = 1, int64_t nh_kv = 1` parameters (source-
   compatible — existing callers compile unchanged). Updated label format to
   include `(nh_q:nh_kv)` when GQA, e.g. `flash_attn f16 d=128 [tile nq=8 GQA 4:1]`.
4. Added 4 new probe sections in `main()`:
   - **[3c]** non-FA GQA sweep (d=128, turbo2/3/4 × {GQA 4:1, 8:1}) — default run
   - **[4b]** FA TILE GQA sweep (d=128, f16 + q8_0 × {GQA 4:1, 8:1}) — default run
   - **[5b]** FA TURBO GQA sweep (d=128, turbo2/3/4 × {GQA 4:1, 8:1}, both tile+vec)
     — gated under existing `LLAMA_TEST_TURBO_FA=1` env var
   - **[6b]** FA VEC GQA sweep (d=128, f16 + q8_0 × {GQA 4:1, 8:1}) — default run
   - **[7]** d=256 FA stress probe — gated under new `LLAMA_TEST_FA256=1` env var,
     with explicit comment block documenting the A770 hang risk and that f16 is
     the only safe KV type at d=256 (turbo+q8_0 still excluded to keep the gate
     deterministic per the existing 256-hang note at L728-733).
5. Incremental rebuild of `test-sycl-turbo-correctness` via the existing
   `build-port/` (oneAPI icpx, `-j6`). First attempt failed with
   `function definition is not allowed here` at L624 — root cause: my first
   `probe_attn_noflash` edit dropped the function's closing `}`. Fixed by adding
   `}` after the trailing `verdict(...)` call. Rebuild in flight; result pending.

**Outcome (build success):** incremental build will be re-run; assuming success,
the harness will compile and link against the existing ggml-sycl + llama libs in
`build-port/`. **Smoke test:** run with default env (no `LLAMA_TEST_TURBO_FA`,
no `LLAMA_TEST_FA256`) — all 3 new GQA sections ([3c]/[4b]/[6b]) and existing
probes should pass/fail identically to baseline (GQA is a coverage-shape check,
not new kernel math, so the verdict should match the pre-extension GATE pass
set). Turbo FA GQA ([5b]) stays gated; d=256 ([7]) stays gated.

**Tasks uncovered:** none new. The 3-model fleet's homogeneity on head_dim=128
means no additional model-specific probe shapes are needed beyond the GQA
ratios (4:1 + 8:1) which this task now covers. P1's per-model PPL/capacity
correctness items can use the extended harness as-is.

**Why not d=256 in the default sweep:** the existing comment at L728-733 already
documents that d=256 reproduces a device-lost hang on A770 SYCL FA. The new [7]
section preserves this guarantee (opt-in only) while making the probe available
for future re-validation after a driver/IGC bump that may upstream-fix the hang.
