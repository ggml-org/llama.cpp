---
name: perf-optimization-workflow
description: >
  Systematic workflow for optimizing llama.cpp performance on Vulkan/GPU backends.
  Use this skill whenever starting a new optimization task, investigating a performance
  regression, or planning a new kernel/fusion. Covers the full cycle from clearing
  blockers, understanding architecture, profiling, specifying, implementing, measuring,
  and deciding to keep or revert. Also use when the user says "optimize", "make it faster",
  "profile", "benchmark", or asks about the optimization process.
---

# Performance Optimization Workflow for llama.cpp

A battle-tested workflow for optimizing llama.cpp inference speed, developed through
multiple optimization cycles on AMD Strix Halo (Vulkan backend). Applies broadly to
any GPU backend optimization work.

## The Workflow

### Step 0: Clear blockers before starting

Before writing any code or running any benchmark, ensure the environment is clean:

- **Kill ALL llama processes**: `taskkill /f /im llama-cli.exe`, `taskkill /f /im llama-bench.exe`, and `taskkill /f /im llama-server.exe`. Running processes lock DLLs and cause `LNK1104: cannot open llama.dll` linker errors. Benchmarks also spawn llama processes that persist after interruption. Always kill before building AND before benchmarking.
- **Remove stale DLLs**: `rm -f bin/llama.dll` if the lock persists after killing processes.
- **Check GPU occupancy**: Look in Task Manager for zombie GPU processes holding Vulkan device locks or skewing benchmark results.
- **Set build environment**: On Windows, `vcvarsall.bat` fails from bash — set `INCLUDE`/`LIB` manually or use a build batch file.
- **Touch shader-gen stamps**: Skip the shader sub-build when only changing C++ code (avoids minutes of unnecessary recompilation from ExternalProject include path issues).
- **Know your compiler pitfalls**: e.g., avoid `std::chrono` in ggml-vulkan.cpp on MSVC (use `ggml_time_us()` instead).

### Step 1: Understand the architecture visually

Before optimizing, build a clear mental model using Mermaid diagrams:

- **State machine diagrams**: Map the lifecycle of dispatches, batch accumulation, fusion detection, command buffer recording.
- **Flowcharts**: Decision trees for op eligibility, dependency checks, hazard detection (RAW/WAR).
- **Sequence diagrams**: CPU-GPU interaction timelines, barrier insertion, buffer synchronization.
- **Data flow diagrams**: Which ops feed into which, where dependency chains exist, what can run in parallel vs. what is sequential.

This is how we:
- Discovered SSM dependency chains that made REPEAT batching impossible (flowchart showed every op reads the previous op's output)
- Designed the batched elementwise state machine correctly on the first attempt (state diagram caught edge cases before implementation)
- Understood why ADD batching was net-negative (sequence diagram showed flush overhead exceeding dispatch savings)

**Update diagrams as the architecture evolves** — they are living documentation, not one-time artifacts.

### Step 2: Profile to find the bottleneck

Use real instrumentation, never assumptions:

- **`GGML_VK_PERF_LOGGER`**: Per-op, per-barrier, per-dispatch timing on Vulkan
- **CPU timing**: `ggml_time_us()` around graph processing to measure host-side overhead
- **Categorize**: Group results into matmuls, small ops, barriers, CPU processing — understand what percentage each category consumes

Key questions to answer:
- What is the total time per token?
- What percentage is matmul (bandwidth-bound) vs. dispatch overhead?
- How many dispatches per token? How many barriers?
- Where are the dependency chains?

### Step 3: Calculate the theoretical gain before writing code

For every proposed optimization:

1. **Count the ops it affects** — how many dispatches/barriers will be removed?
2. **Estimate the time savings** — at ~5-6 us per dispatch, removing N dispatches saves ~N*5.5 us
3. **Check dependency chains** — can the ops actually be combined, or are they sequentially dependent (RAW hazards)?
4. **Verify mathematical semantics** — if substituting ops (e.g., replacing MUL+SUM_ROWS with MUL_MAT), verify the math is equivalent. `ggml_mul_mat` computes A^T @ B, NOT A @ B.
5. **Calculate the ceiling** — what is the maximum possible gain? Is it worth the implementation complexity?

If the theoretical gain is <2% or the dependency analysis shows ops can't combine, stop here.

### Step 4: Load or create specialized agents and skills

Before diving into implementation, assemble a team of specialists:

- **Load existing skills**: Check `.claude/skills/` for relevant skills (e.g., `vulkan-dispatch-optimization`, `llama-cpp-vulkan-internals`, `strix-halo-optimizer`). These contain hard-won knowledge from previous optimization cycles.
- **Create new skills** if the task touches a domain not yet covered — encode the knowledge so it persists across sessions.
- **Spin up subagents** for parallel research — each agent focuses on one concern:
  - One agent investigates the shader code
  - One agent traces the op dependency graph
  - One agent checks for existing fusions or optimizations
  - One agent reviews the model architecture for structural patterns
  - One agent profiles and benchmarks
- **Work as a team**: Agents report findings back, and the main context synthesizes a plan from their parallel research. This is faster than sequential investigation and protects the main context window from being overwhelmed.
- **Update skills after the work**: When an optimization succeeds or a dead end is confirmed, update the relevant skill files so future sessions start with that knowledge.

### Step 5: Research what exists online

Before implementing from scratch, search for prior art:

- **WebSearch / WebFetch**: Search for papers, blog posts, forum threads on the specific technique (e.g., "fused SSM recurrence kernel Vulkan", "Delta-Net inference optimization", "Mamba GPU kernel fusion")
- **Hugging Face**: Check if reference implementations exist in PyTorch/Triton that can inform the GLSL design
- **GitHub issues/PRs**: Search llama.cpp issues and competing projects (vLLM, TensorRT-LLM, GGML forks) for similar optimizations
- **arXiv papers**: The original Delta-Net / Mamba papers often include kernel implementation details

This avoids reinventing the wheel and may reveal pitfalls others already discovered.

### Step 6: Spec before code

Write specifications before implementing:

- **Gherkin scenarios**: Define expected behavior, edge cases, and exclusions (e.g., "SIGMOID is NOT batched when part of topk_moe fusion")
- **State machine diagrams**: Formalize the accumulation/flush/dispatch lifecycle
- **Dependency check flowcharts**: Document hazard detection logic
- **Flush sequence diagrams**: Specify the exact barrier/memcpy/dispatch order

This catches bugs before they happen. The batched elementwise mega-kernel was implemented correctly on the first pass because the spec caught WAR hazards, batch-full conditions, and fusion exclusions upfront.

### Step 7: Cross-verify with a second opinion

Use the Copilot CLI to get independent verification from multiple models:

```bash
# GPT-5.4 for broad reasoning and architectural review
copilot --model gpt-5.4 -p "your question" --allow-all-tools

# GPT-5.3 Codex for code-focused analysis (shader logic, op semantics, implementation review)
copilot --model gpt-5.3-codex -p "your question" --allow-all-tools
```

Query both models when the stakes are high (e.g., op substitutions, shader correctness). If they disagree, investigate further before proceeding.

Use this for:
- **Verifying mathematical semantics** — "Does ggml_mul_mat compute A^T@B or A@B?" before committing to an op substitution
- **Checking architectural assumptions** — "In this SSM recurrence, can these ops be reordered or are they sequentially dependent?"
- **Reviewing optimization proposals** — share profiling data and ask if the proposed approach makes sense
- **Sanity-checking shader logic** — paste a GLSL snippet and ask for correctness review (GPT-5.3 Codex excels here)
- **Catching blind spots** — different models may spot issues or suggest approaches you haven't considered

Keep prompts focused and specific. Include relevant code snippets or profiling data in the prompt for accurate answers.

### Step 8: Small, isolated changes with immediate measurement

- **One change at a time**: Never combine multiple optimizations in one build
- **Warm up the GPU first**: Cold-start runs can be ~24% slower on RDNA 3.5 iGPU. Run a short tg16 benchmark first and discard the result before the real measurement.
- **Benchmark immediately**: Run the model right after building — don't batch multiple changes before measuring
- **Record exact parameters**: Always note -n (sequence length), -r (repeats), -p (prompt length). A "regression" from 58→44 tok/s once turned out to be tg64 vs tg128.
- **Compare against baseline**: Keep a known-good build (e.g., `build-master/`) for A/B comparison
- **Self-contained changes**: Each optimization should be independently revertable

### Step 9: Commit improvements immediately

When a change produces a measurable tok/s improvement:

- **Commit right away** with a descriptive message including the before/after numbers
- **Tag the commit** with the optimization name for easy reference
- This locks in gains and provides a known-good restore point if future changes regress
- Format: `"vulkan: <what changed> (+X% tg, before→after tok/s)"`

If the change shows no improvement or regression, revert instead of committing.

### Step 10: Revert fast, don't debug hopeless approaches

- If the numbers don't improve, **understand why** (dependency chains? semantic mismatch? flush overhead?)
- **Revert in minutes**, not hours — `git checkout HEAD -- <file>` and move on
- Don't patch a fundamentally flawed approach (e.g., using `goto` to skip state machine setup)
- Keep a log of what was tried and why it failed — this prevents revisiting dead ends

### Step 11: Update or create skills after every optimization cycle

After each optimization attempt (success or failure), encode what was learned:

- **Success**: Update the relevant skill with the technique, numbers, and any gotchas discovered
- **Failure / dead end**: Add the approach to the skill's anti-patterns or "dead ends" section so future sessions don't retry it
- **New domain**: If the optimization touched a domain without a skill (e.g., a new model architecture, a new backend feature), create a new skill under `.claude/skills/<name>/SKILL.md`
- **Update MEMORY.md**: Record new baseline numbers, new bottleneck breakdown, and any changed architectural understanding
- **Skills to check/update after each cycle**:
  - `perf-optimization-workflow` — add new anti-patterns or workflow refinements
  - `vulkan-dispatch-optimization` — new fusion techniques or dead ends
  - `vulkan-moe-bandwidth-analysis` — updated bandwidth measurements
  - `strix-halo-optimizer` — hardware-specific findings
  - `ggml-op-development` — if a new op was added, document any pitfalls encountered
  - `build-troubleshooting` — if new build issues were hit and solved

This is how institutional knowledge compounds across sessions. A skill that saves 10 minutes per session pays for itself after 2-3 uses.

## Anti-Patterns to Avoid

| Anti-Pattern | Example | Why It Fails |
|---|---|---|
| Optimizing without profiling | "Let's batch all small ops" | Some ops can't batch due to dependencies |
| Substituting ops without verifying semantics | MUL_MAT for MUL+SUM_ROWS | ggml_mul_mat computes A^T@B, not A@B |
| Assuming batchability without checking deps | REPEAT op batching | Ops are isolated by sequential RAW hazards |
| Using goto to skip complex state machines | Fusion plan cache | Skips essential state setup, causes regressions |
| Not measuring after each change | Stacking 3 optimizations before benchmarking | Can't tell which one helped or hurt |
| Not clearing system resources first | Building with llama-cli.exe still running | LNK1104 linker errors, wasted time |
| Debugging a dead approach too long | Spending hours fixing fusion cache goto | Should have reverted and tried a different design |

## Quick Reference: One-Sentence Summary

**Clear blockers, diagram the architecture, profile the bottleneck, calculate the ceiling, spec the solution, implement the smallest change, measure immediately, revert without hesitation if it doesn't work, and always update skills with what you learned.**
