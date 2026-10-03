# Fork, build, run, and maintain the project

Status: operator guide for the active patchset. Stock Vulkan commands, the initial benchmark runner, and its first IQ3_S baseline are complete; Vulkan optimization modules remain unimplemented.
Specification date: 2026-10-01.

## 1. Environment assumptions

The reference deployment is Fedora Linux 44 KDE with kernel `7.2.8-200.fc44.x86_64`, one RX 7900 XTX, Vulkan, and Mesa RADV `26.2.3`. The Linux package example below uses Debian/Ubuntu names; use Fedora-equivalent packages on the reference machine. A Windows Vulkan build is documented separately, and requires independent performance validation with its driver.

You need Git, a C/C++ toolchain, CMake, Ninja, Vulkan development libraries, the `glslc` shader compiler, SPIR-V headers, and a working Vulkan driver. Python is needed to develop/package the benchmark; use only the project-local `.venv-bench` virtual environment for its dependencies. The eventual packaged executable is intended to run without a separate Python installation.

Do not install ROCm merely for this project. Do not use HIP build options: the chosen backend is Vulkan.

## 2. Fork and clone

In GitHub, open [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and choose **Fork** into your account. Keep the fork name `llama.cpp` unless you prefer another name. The commands below assume that name. Replace `YOUR_GITHUB_USER` with your username.

```bash
git clone https://github.com/YOUR_GITHUB_USER/llama.cpp.git
cd llama.cpp
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git remote -v
git fetch upstream
git switch master
git merge --ff-only upstream/master
git push origin master
git switch -c rx7900xtx-vulkan
```

If your fork's master has unexpected commits and cannot fast-forward, inspect them before deciding how to reconcile it. Do not reset away work blindly.

Set your commit identity if not already configured:

```bash
git config user.name "Your Name"
git config user.email "YOUR_COMMIT_EMAIL"
```

Copy the project Markdown documents into the repository root. Add them and commit:

```bash
git add goal.md architecture.md benchmark.md howtobuild.md progress.md
git commit -m "docs: define RX 7900 XTX MTP optimization project"
git push -u origin rx7900xtx-vulkan
```

Branch roles:

| Branch | Purpose |
|---|---|
| `master` | Mirror upstream; no optimization commits |
| `rx7900xtx-vulkan` | Small accepted patchset and project documentation |
| `experiment/<name>` | Temporary hypothesis and measurements |
| `backup/<name>` | Recovery reference before rebase or history editing |

Do not merge unreviewed experiments into the accepted patch branch.

## 3. Linux dependencies and device checks

For Debian/Ubuntu, install the toolchain and typical Vulkan packages:

```bash
sudo apt-get update
sudo apt-get install git build-essential cmake ninja-build libvulkan-dev glslc spirv-headers vulkan-tools mesa-vulkan-drivers python3 python3-venv
```

Package versions must satisfy the actual checked-out CMake requirements. If your distribution's headers/compiler are too old, use a suitable Vulkan SDK according to upstream instructions rather than mixing incompatible files. Installing development packages does not guarantee the runtime selects RADV.

Verified reference inventory: llama-server build `0.5.0-dev-11380`, source commit `0f0796f9076f3ebd3f49fa9be74598b33c42aada`, GCC `16.2.1`, CMake `4.3.0`, shaderc `2026.1`, Ryzen 5 7600X, and approximately 30 GiB RAM. `llama-server --list-devices` reports the RX 7900 XTX as `Vulkan0`; do not use the integrated Radeon GPU reported as `Vulkan1`.

Check the environment:

```bash
cmake --version
ninja --version
c++ --version
glslc --version
vulkaninfo --summary
git rev-parse HEAD
```

Inspect the RX 7900 XTX entry and driver name/version. If several Vulkan devices or drivers are installed, verify llama-server selects the intended GPU/driver. Do not hardcode an ICD path from another machine. Record the effective setup in benchmark manifests.

## 4. First build: unmodified Vulkan behavior

Before the project option exists, build an ordinary Vulkan control:

```bash
cmake -S . -B build-control -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-control --parallel 8
./build-control/bin/llama-server --help
./build-control/bin/llama-server --list-devices
```

Adjust parallel jobs for your RAM and CPU. Shader compilation can consume substantial memory; lower the count if the build is killed. Do not assume configuring successfully means Vulkan was found: inspect CMake output and the executable's device list.

The source commit selected for initial work must support the **actual inspected GGUF and embedded MTP**. If it does not, determine the necessary upstream support before benchmarking. These documents do not certify that every master revision loads the supplied filenames.

## 5. Paired builds after the optimization option is implemented

The following option is **ours**, not upstream: `GGML_VULKAN_RX7900XTX_OPT`. Until the patch implements it, CMake may warn that the variable is unused. Such a warning means the build is not an optimization build.

Build both variants from the same source tree/commit:

```bash
cmake -S . -B build-control -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_VULKAN_RX7900XTX_OPT=OFF
cmake --build build-control --parallel 8
cmake -S . -B build-rx7900xtx -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_VULKAN_RX7900XTX_OPT=ON
cmake --build build-rx7900xtx --parallel 8
```

Keep compiler, dependencies, CPU optimization flags, and all unrelated CMake settings identical. The only intended difference is the project option. Save the CMake cache and build identity with benchmark evidence. MTP is enabled at runtime in both variants.

For incremental builds:

```bash
cmake --build build-rx7900xtx --parallel 8
cmake --build build-control --parallel 8
```

After changing toolchains or major dependencies, configure new build directories instead of reusing an incompatible cache. Never modify generated shader headers manually.

## 6. Inspect and run the real models

Keep weights outside the Git checkout. The verified primary artifact is `/home/tmatkovic/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`; it is 12120016896 bytes with SHA-256 `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`. Record checksums once when an artifact changes:

```bash
sha256sum /home/tmatkovic/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
```

Use the future harness inventory functionality or the pinned upstream GGUF tooling to inspect architecture metadata, tensor types/shapes, embedded MTP tensors, and context-related metadata. The primary GGUF has `general.architecture = qwen35`, 866 tensors, and declares `qwen35.nextn_predict_layers`; do not assume uniform IQ3_S quantization throughout it. The Q4_K_M model is out of scope and must not be inspected or benchmarked for this project.

Read binary help before copying the examples. The current upstream server reference documents the MTP spellings used here; flags may change. `n-max=4` below is only an initial example, not a measured best value. Confirm any model-specific draft loading/setup requirements in the selected revision.

Illustrative IQ3_S launch:

```bash
./build-rx7900xtx/bin/llama-server -m /home/tmatkovic/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -c 200000 -np 1 -ngl all --spec-type draft-mtp --spec-draft-n-max 4 --host 127.0.0.1 --port 8080
```

Use `build-control` before an optimized binary exists. Run only one server at a time. Confirm effective capacity, single slot, GPU placement, and active MTP from startup/request behavior. A filename containing `mtp` or a successful HTTP response is insufficient proof of active speculation.

Batch/ubatch, cache types, flash attention, CPU threads, and real sampling settings must be established from the user's environment, then pinned in local benchmark config. Do not add context-extension/RoPE flags without verifying the real model's requirements. Large configured capacity does not itself establish model quality at that depth.

After runtime controls are implemented, this should disable our patches while retaining upstream MTP:

```bash
GGML_VULKAN_RX7900XTX_OPT=off ./build-rx7900xtx/bin/llama-server -m /home/tmatkovic/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -c 200000 -np 1 -ngl all --spec-type draft-mtp --spec-draft-n-max 4
```

Use compiled ON/OFF variants for primary acceptance evidence. Runtime-off is convenient for diagnostics and checking gate coverage.

## 7. Build and run the future benchmark executable

The baseline benchmark package and CLI are implemented. Create the virtual environment, then install the package in editable mode:

```bash
python3 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -e ./tools/rx7900xtx-bench
python -m rx7900xtx_bench --help
```

Create and activate `.venv-bench` before every source-development session. Do not use the system Python or globally installed Python packages for the benchmark. The virtual environment is local build state and must stay ignored by Git. Packaging as a standalone executable remains future work.

Create a machine-local `bench.local.json` with model/server paths and the verified runtime settings. Keep it ignored by Git. Run a dry run, then the baseline:

```bash
python -m rx7900xtx_bench --preset qwen35-iq3s --config bench.local.json --dry-run
python -m rx7900xtx_bench --preset qwen35-iq3s --config bench.local.json --depth 4096 --runs 5 --label initial-control
```

See `benchmark.md` for complete CLI semantics, MTP sweeps, paired builds, fresh/reused prompts, result files, and full validation capped at 96k. See `progress.md` for the initial reproducible baseline. No kernel optimization is accepted before reproducible MTP measurements exist.

## 8. Experiment workflow

Start from a clean accepted branch. A separate worktree avoids mixing experiments into your main checkout:

```bash
git switch rx7900xtx-vulkan
git status
git worktree add ../llama-iq3s-experiment -b experiment/iq3s-variant rx7900xtx-vulkan
```

Build and benchmark inside that worktree. Both A/B variants must use that experiment's source commit. First check affected kernel/dispatch correctness, then quick MTP results, then common/full results if promising.

Record the result in `docs/rx7900xtx/experiments/`. Clean the change into reviewable commits. To promote a measured patch, return to the accepted checkout and cherry-pick the exact reviewed commit:

```bash
git switch rx7900xtx-vulkan
git cherry-pick EXPERIMENT_COMMIT_SHA
git push origin rx7900xtx-vulkan
```

Replace `EXPERIMENT_COMMIT_SHA` with the real commit. If several dependent commits are needed, preserve their order and verify the resulting state. Do not cherry-pick unrelated exploratory changes.

For a rejected experiment, preserve useful notes/results outside the disposable worktree before removing it. Removing a worktree containing uncommitted work is not part of this routine guide.

## 9. Rebase onto upstream

Rebase changes commit IDs. Since you are the sole developer, rewriting the optimization branch is appropriate when done deliberately. Start with a clean working tree and a backup branch. If dirty, commit suitable work or deliberately stash it; do not hide changes automatically.

```bash
git switch rx7900xtx-vulkan
git status
git fetch origin
git fetch upstream
git branch backup/rx7900xtx-before-rebase-2026-10-01
git log --oneline --left-right HEAD...origin/rx7900xtx-vulkan
git merge-base HEAD upstream/master
git rebase upstream/master
```

Use a unique backup name each time. Inspect the left/right log for unexpected remote work. Record the old base SHA and accepted tip before rebasing.

If there are conflicts:

1. Read `docs/rx7900xtx/upstream-hooks.md` and the upstream change.
2. Preserve the intended new upstream behavior and adapt the small hook.
3. Stage only resolved files with `git add PATH`.
4. Run `git rebase --continue`.
5. Use `git rebase --abort` if you need to return to the pre-rebase state.

Never resolve a semantic conflict by blindly choosing all “ours.” Upstream may have fixed the same operation or introduced a faster path; remove or revise redundant patches when evidence supports it.

After rebase, rebuild ON and OFF, run affected correctness checks, confirm MTP on IQ3_S, and refresh the control benchmark. Compare the new ON/OFF pair at the new base. Do not attribute a change against an old-base result solely to our patchset.

Review the rebased diff and update status/hook notes. Then push using a lease, not an unconditional force:

```bash
git push --force-with-lease origin rx7900xtx-vulkan
git switch master
git merge --ff-only upstream/master
git push origin master
git switch rx7900xtx-vulkan
```

If the lease fails, inspect remote changes before retrying. Keep the backup until the new branch is verified. Rebase regularly, but do not invalidate a live experiment's baseline midway through its measurements.

## 10. Test and release checklist

- Build project ON and OFF from the same source state.
- Verify an ordinary upstream build remains usable and OFF preserves its behavior.
- Run affected backend correctness/state tests; inspect actual test target/help names at the pinned revision.
- Verify fallback on unsupported types/layouts and project runtime-off.
- Run common MTP comparisons; run full validation before retaining a substantial optimization.
- Record exact model hashes, effective settings, source and binary identities, and result paths.
- Test the standalone benchmark executable and owned-process cleanup.
- Update `status.md`, hook documentation, and experiment decision notes.

Use CTest when the configured build supplies applicable tests:

```bash
ctest --test-dir build-rx7900xtx -N
ctest --test-dir build-rx7900xtx --output-on-failure
```

Run the benchmark harness unit tests from the activated `.venv-bench` environment:

```bash
python -m unittest discover -s tools/rx7900xtx-bench/tests -v
```

Listing/tests depend on the upstream test options and project test registration. An empty CTest suite is not correctness validation; configure the required tests and run the relevant backend tool directly if necessary.

## 11. Windows alternative

Install Git, Visual Studio 2022 with Desktop development with C++, CMake/Ninja, the Vulkan SDK, and a working AMD Vulkan driver. Use **Developer PowerShell for VS 2022** so the compiler is available. Check `glslc`, `cmake`, and Vulkan device enumeration before building.

The fork/rebase commands are the same. With Ninja:

```powershell
cmake -S . -B build-control -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-control --parallel 8
.\build-control\bin\llama-server.exe --help
```

After our CMake option exists, add `-DGGML_VULKAN_RX7900XTX_OPT=OFF` to control and configure a separate `build-rx7900xtx` with `ON`. For Visual Studio generators, use `--config Release` and expect executables under `bin\Release` instead. Do not mix paths from these two generator layouts.

Hash a model with:

```powershell
Get-FileHash 'D:\models\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' -Algorithm SHA256
```

Use the same server arguments as Linux, with quoted Windows paths. For runtime-off diagnostics after implementation:

```powershell
$env:GGML_VULKAN_RX7900XTX_OPT = 'off'
```

Remove the environment setting after that diagnostic session. Windows uses its selected Vulkan driver; Linux/RADV results do not establish Windows performance. If Windows is the actual deployment, make its measured driver/device capabilities the reference before selecting tuning rules.

## 12. Files to keep and troubleshooting

Keep models, build directories, benchmark virtual environments, distribution output, raw results, and machine-local config ignored. Add project-specific patterns to `.gitignore` or `.git/info/exclude` without removing upstream ignore rules. Keep selected experiment summaries and inventory manifests under version control; inventories must not contain private paths unnecessarily.

| Symptom | First check |
|---|---|
| Vulkan not found | Loader headers/libraries, SDK environment, CMake output |
| `glslc` missing | Shader compiler installation and PATH |
| SPIR-V header failure | `spirv-headers` package or SDK include directory |
| Shader edits do not rebuild | CMake/generator dependencies for nested project shaders and includes |
| Project option reported unused | Patch implementing the option is absent or not wired to this build |
| Wrong GPU/driver | Device list, selected Vulkan device, effective ICD environment |
| MTP inactive | Real GGUF head/support, pinned upstream implementation, effective arguments and request counters |
| Model-load failure | GGUF metadata and upstream compatibility; do not rename/substitute the model |
| A/B results incompatible | Model/source/runtime/prompt provenance and resolved settings |
| Slow prefix benchmark | Verify actual reuse/newly evaluated token counts and state handling |

These documents record an initial verified IQ3_S MTP baseline; see `progress.md`. They do not claim that a Vulkan optimization has been implemented or accepted.

## Sources and update policy

- [Official build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/howtobuild.md)
- [Official server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Vulkan CMake integration](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/CMakeLists.txt)

The official sources were checked while preparing this guide. Pin a real upstream commit for work and recheck its local help/CMake/source before running commands; master is moving. Project-owned flags and tool interfaces are explicitly marked above.
