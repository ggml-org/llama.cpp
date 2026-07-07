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