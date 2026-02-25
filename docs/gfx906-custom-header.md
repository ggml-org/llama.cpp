# gfx906 Custom Header

> [!IMPORTANT]
> Canonical branch header for `gfx906` repo-specific changes.
> Keep this block current when any custom change or benchmark result changes.
>
> Last updated: 2026-02-25 21:51 UTC  
> Branch head: `89a3923d`  
> Base reference: `origin/master @ a96a1120`

Status legend:
- `upstreamed (PR #N)` = merged upstream
- `not upstreamed (PR #N)` = PR exists but not merged
- `not upstreamed` = no PR yet

| ID | Change | Scope | Upstream State | Benchmarks |
| --- | --- | --- | --- | --- |
| GFX906-001 | Fast mmap GPU loading path: staged pinned host ring, async uploads, parallel mmap context loading, and no `MAP_POPULATE` for GPU offload. Includes `LLAMA_LOAD_N_BUFFERS` and `LLAMA_LOAD_BUFFER_MB` tuning knobs. | `src/llama-model.cpp`, `src/llama-model-loader.cpp`, `src/llama-model-loader.h` | `not upstreamed (PR #1 OPEN)` https://github.com/skyne98/llama.cpp-gfx906/pull/1 | Qwen3.5-122B-A10B Q4_K_XL, 4x MI50, full context: baseline mmap `17379.03 ms`; PR default `7808.37 ms` (`-55.1%`); PR tuned (`8x32`) `7645.37 ms` (`-56.0%`); prior direct-io `11445.16 ms` |
