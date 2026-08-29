<p align="center">
  <a href="#中文">中文</a> · <a href="#english">English</a>
</p>

\---

# 中文

## LLaMA.cpp — Qwen4Exp 修复版

基于 [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) `qwen4exp/qwen3.8-flash-next` 分支，手动合并了两个上游 PR 的修复。

### 为什么需要手动整合？

上游的两个 PR 分别解决了不同方面的问题，但尚未合并到主分支：

|PR|状态|内容|
|-|-|-|
|[#27879](https://github.com/ggml-org/llama.cpp/pull/27879)|Open (Draft)|Qwen4Exp 正确性修复|
|[#27836](https://github.com/ggml-org/llama.cpp/pull/27836)|Open|Qwen4Exp MTP draft head 支持|

**PR #27879** 修复了 8 个正确性问题（稀疏注意力、PLE 嵌入宽度、元数据校验、QSA 索引器、回滚状态、tensor-split 拒绝、Metal 对齐/索引），这些是**运行时必须的修复**——不打这个补丁，长上下文、多序列、投机解码等场景会出现静默错误或崩溃。

**PR #27836** 添加了 NextN/MTP draft head 支持（`--spec-type draft-mtp`），这是**功能增强**——让 Qwen3.8-Flash-Next 能利用 MTP 头进行投机解码，显著提升推理速度。

两个 PR 有代码冲突（都改了 `qwen4exp.cpp`），无法简单 cherry-pick。本项目手动解决了冲突，将两个修复合并到一起编译。

### 合并过程与冲突解决

#### 第一步：应用 PR #27879（正确性修复）

```bash
git apply --exclude='tests/\*' pr27879.diff
```

顺利应用。排除 `tests/` 目录是因为 unsloth 分支的测试文件结构与上游不同，测试代码不影响运行时功能。

#### 第二步：应用 PR #27836（MTP 支持）

```bash
git apply pr27836.diff
# error: patch failed: src/models/qwen4exp.cpp:155
# error: src/models/qwen4exp.cpp: patch does not apply
```

**冲突原因**：两个 PR 都修改了 `src/models/qwen4exp.cpp` 的 `load\_arch\_tensors` 函数。PR #27879 把函数里的 `0` 改成了 `flags`（引入了 `mtp\_flags` 变量），而 PR #27836 在同一个位置添加了 MTP head 的 tensor 创建代码。Git 无法自动判断如何合并。

#### 第三步：使用 `--reject` 模式部分应用

```bash
git apply --reject pr27836.diff
```

结果：

* 8/9 个 hunk 成功应用（`conversion/qwen4exp.py`、`gguf-py/`、`src/llama-arch.cpp`、`src/llama-arch.h`、`src/llama-model.cpp`、`src/llama-model.h`、`src/models/models.h`）
* 1 个 hunk 被拒绝：`src/models/qwen4exp.cpp` 的 hunk #3

生成了 `.rej` 文件，记录了未应用的改动。

#### 第四步：手动合并冲突

冲突的核心是 `qwen4exp.cpp` 中 `load\_arch\_tensors` 函数的 tensor 创建代码。PR #27879 把所有 `create\_tensor(..., 0)` 改成了 `create\_tensor(..., flags)`，而 PR #27836 在函数末尾添加了 MTP head 的 tensor 创建。

**解决方案**：

1. **全局替换**：将 `load\_arch\_tensors` 函数内所有 `, 0)` 替换为 `, flags)`
2. **修复越界**：发现函数外的 `tok\_embd`、`hc\_head\_\*` 等也被误替换（它们不在 `flags` 变量的作用域内），手动改回 `, 0)`
3. **插入 MTP head 代码**：在 `load\_arch\_tensors` 函数末尾（`ffn\_down\_shexp` 之后）插入 PR #27836 的 MTP head tensor 创建代码：

   * `nextn.enorm`、`nextn.hnorm`、`nextn.eh\_proj`
   * `nextn.hc\_head\_norm`、`nextn.hc\_head\_down`、`nextn.hc\_head\_up`
   * `nextn.embed\_tokens`、`nextn.shared\_head\_head`（标记为 `TENSOR\_NOT\_REQUIRED`）
4. **更新 `build\_arch\_graph`**：添加 MTP 图类型判断

#### 第五步：验证编译

```bash
cmake --build build  # 259 targets, 全部成功
```

#### 冲突文件清单

|文件|PR #27879 改动|PR #27836 改动|解决方式|
|-|-|-|-|
|`src/models/qwen4exp.cpp`|`0` → `flags` + 校验逻辑|MTP head tensor + graph\_mtp|手动合并|
|`src/llama-arch.cpp`|新增 arch 常量|新增 MTP arch 常量|自动合并|
|`src/llama-arch.h`|新增 arch 枚举|新增 MTP 枚举|自动合并|
|`src/llama-model.cpp`|小改动|小改动|自动合并|
|`src/llama-model.h`|无|新增 MTP 字段|自动合并|
|`src/models/models.h`|小改动|小改动|自动合并|

### 目录结构

```
llamacpp/
├── NOTES.md           # 本文档
├── patches/            # 上游 PR 的 diff 文件（本地存档）
│   ├── pr27879.diff    # 正确性修复
│   └── pr27836.diff    # MTP 支持
└── src/                # 完整源码（已合并两个 PR 的修复，3471 个文件）
    ├── CMakeLists.txt  # 顶层构建配置
    ├── src/            # 核心 C++ 源码（llama.cpp、llama-model.cpp 等）
    ├── ggml/           # ggml 张量计算后端
    ├── include/        # 公共头文件
    ├── tools/          # llama-server、llama-cli 等可执行工具
    ├── common/         # 公共库（llama-common）
    ├── conversion/     # 模型转换工具（convert\_hf\_to\_gguf.py 等）
    ├── examples/       # 示例代码
    ├── tests/          # 测试代码
    └── ...
```

### 编译环境

* **OS**: Windows 11
* **CUDA**: 13.3
* **Compiler**: MSVC 19.44 (Visual Studio 2022 Build Tools)
* **Build system**: CMake + Ninja
* **GPU**: NVIDIA (Compute Capability 8.6)

### 从源码编译

```bash
# 直接编译（已包含合并后的修复）
cmake -B build -G Ninja -DGGML\_CUDA=ON -DCMAKE\_BUILD\_TYPE=Release
cmake --build build

# 或者从零开始：克隆 + 手动合并
git clone --branch qwen4exp/qwen3.8-flash-next https://github.com/unslothai/llama.cpp.git
cd llama.cpp
git apply ../patches/pr27879.diff --exclude='tests/\*'
git apply ../patches/pr27836.diff
cmake -B build -G Ninja -DGGML\_CUDA=ON -DCMAKE\_BUILD\_TYPE=Release
cmake --build build
```

### PR #27879 详细内容

由 [tarruda](https://github.com/tarruda) 提交，使用 GPT 5.6 Sol 辅助完成。修复内容：

1. **稀疏注意力块选择** — QSA 块按逻辑 token 顺序构建，避免长上下文/多序列推理的 logit 偏差
2. **PLE 嵌入宽度独立支持** — 不再假设 PLE KV 投影等于主隐藏宽度
3. **元数据加载校验** — GDN、hyper-connection、QSA、压缩、PLE 不变量检查
4. **QSA 索引器缓存复制** — 序列 fork 时正确复制索引器 key
5. **回滚状态启用** — 投机解码的 GDN/PLE 状态能正确恢复
6. **拒绝 tensor-split** — 防止不支持的模式导致静默错误
7. **Metal 对齐修复** — 动态线程组内存 16 字节对齐
8. **Metal 索引拓宽** — 16-bit → 32-bit，大 batch 不溢出

### PR #27836 详细内容

由 [rmonsurate](https://github.com/rmonsurate) 提交。添加内容：

* NextN/MTP draft head 支持（`--spec-type draft-mtp`）
* 转换器支持（`--mtp` 参数）
* 在 M3 Max 128GB 上测试通过，UD-IQ4\_XS 量化输出与官方一致

### License

与上游 [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) 相同。

### AI 使用声明

本项目在开发过程中使用了 AI 辅助工具，具体如下：

* Mimo AI：协助生成了 README 文档的初稿大纲，并提供了 `git` 仓库克隆和分支切换的命令示例。

所有 AI 生成的建议和代码片段均经过了人工审查、测试和修改，最终代码由人类贡献者负责。

\---

# English

## LLaMA.cpp — Qwen4Exp Patched Build

Based on the [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) `qwen4exp/qwen3.8-flash-next` branch, with two upstream PRs manually merged.

### Why Manual Integration?

Two upstream PRs address different issues but haven't been merged into the main branch yet:

|PR|Status|Description|
|-|-|-|
|[#27879](https://github.com/ggml-org/llama.cpp/pull/27879)|Open (Draft)|Qwen4Exp correctness fixes|
|[#27836](https://github.com/ggml-org/llama.cpp/pull/27836)|Open|Qwen4Exp MTP draft head support|

**PR #27879** fixes 8 correctness issues (sparse attention, PLE embedding width, metadata validation, QSA indexer, rollback state, tensor-split rejection, Metal alignment/indexing). These are **runtime-critical fixes** — without this patch, long-context, multi-sequence, and speculative decoding scenarios produce silent errors or crashes.

**PR #27836** adds NextN/MTP draft head support (`--spec-type draft-mtp`). This is a **feature enhancement** — it lets Qwen3.8-Flash-Next leverage the MTP head for speculative decoding, significantly improving inference speed.

The two PRs have code conflicts (both modify `qwen4exp.cpp`) and cannot be simple cherry-picked. This project manually resolved the conflicts and compiled both fixes together.

### Merge Process \& Conflict Resolution

#### Step 1: Apply PR #27879 (Correctness Fixes)

```bash
git apply --exclude='tests/\*' pr27879.diff
```

Applied cleanly. The `tests/` directory was excluded because the unsloth branch has different test file structure from upstream, and test code doesn't affect runtime behavior.

#### Step 2: Apply PR #27836 (MTP Support)

```bash
git apply pr27836.diff
# error: patch failed: src/models/qwen4exp.cpp:155
# error: src/models/qwen4exp.cpp: patch does not apply
```

**Conflict cause**: Both PRs modify the `load\_arch\_tensors` function in `src/models/qwen4exp.cpp`. PR #27879 changed `0` to `flags` (introducing the `mtp\_flags` variable), while PR #27836 adds MTP head tensor creation code at the same location. Git couldn't auto-merge.

#### Step 3: Partial Apply with `--reject` Mode

```bash
git apply --reject pr27836.diff
```

Results:

* 8/9 hunks applied successfully (`conversion/qwen4exp.py`, `gguf-py/`, `src/llama-arch.cpp`, `src/llama-arch.h`, `src/llama-model.cpp`, `src/llama-model.h`, `src/models/models.h`)
* 1 hunk rejected: `src/models/qwen4exp.cpp` hunk #3

Generated `.rej` files recording the unapplied changes.

#### Step 4: Manual Conflict Resolution

The core conflict was in the tensor creation code within `load\_arch\_tensors` in `qwen4exp.cpp`. PR #27879 changed all `create\_tensor(..., 0)` to `create\_tensor(..., flags)`, while PR #27836 adds MTP head tensor creation at the end of the function.

**Resolution**:

1. **Global replacement**: Replaced all `, 0)` with `, flags)` inside `load\_arch\_tensors`
2. **Fix out-of-scope**: Found that `tok\_embd`, `hc\_head\_\*` etc. outside the function were also replaced (they're not in `flags` variable scope), manually changed back to `, 0)`
3. **Insert MTP head code**: Added PR #27836's MTP head tensor creation at the end of `load\_arch\_tensors` (after `ffn\_down\_shexp`):

   * `nextn.enorm`, `nextn.hnorm`, `nextn.eh\_proj`
   * `nextn.hc\_head\_norm`, `nextn.hc\_head\_down`, `nextn.hc\_head\_up`
   * `nextn.embed\_tokens`, `nextn.shared\_head\_head` (marked as `TENSOR\_NOT\_REQUIRED`)
4. **Update `build\_arch\_graph`**: Added MTP graph type dispatch

#### Step 5: Verify Build

```bash
cmake --build build  # 259 targets, all succeeded
```

#### Conflict File Summary

|File|PR #27879 Changes|PR #27836 Changes|Resolution|
|-|-|-|-|
|`src/models/qwen4exp.cpp`|`0` → `flags` + validation logic|MTP head tensor + graph\_mtp|Manual merge|
|`src/llama-arch.cpp`|New arch constants|New MTP arch constants|Auto-merged|
|`src/llama-arch.h`|New arch enum|New MTP enum|Auto-merged|
|`src/llama-model.cpp`|Minor changes|Minor changes|Auto-merged|
|`src/llama-model.h`|None|New MTP fields|Auto-merged|
|`src/models/models.h`|Minor changes|Minor changes|Auto-merged|

### Directory Structure

```
llamacpp/
├── NOTES.md           # This document
├── patches/            # Upstream PR diff files (local archive)
│   ├── pr27879.diff    # Correctness fixes
│   └── pr27836.diff    # MTP support
└── src/                # Full source code (both PRs merged, 3471 files)
    ├── CMakeLists.txt  # Top-level build config
    ├── src/            # Core C++ source (llama.cpp, llama-model.cpp, etc.)
    ├── ggml/           # ggml tensor computation backend
    ├── include/        # Public headers
    ├── tools/          # llama-server, llama-cli, and other executables
    ├── common/         # Common library (llama-common)
    ├── conversion/     # Model conversion tools (convert\_hf\_to\_gguf.py, etc.)
    ├── examples/       # Example code
    ├── tests/          # Test code
    └── ...
```

### Build Environment

* **OS**: Windows 11
* **CUDA**: 13.3
* **Compiler**: MSVC 19.44 (Visual Studio 2022 Build Tools)
* **Build system**: CMake + Ninja
* **GPU**: NVIDIA (Compute Capability 8.6)

### Building from Source

```bash
# Build directly (already contains merged fixes)
cmake -B build -G Ninja -DGGML\_CUDA=ON -DCMAKE\_BUILD\_TYPE=Release
cmake --build build

# Or start from scratch: clone + manual merge
git clone --branch qwen4exp/qwen3.8-flash-next https://github.com/unslothai/llama.cpp.git
cd llama.cpp
git apply ../patches/pr27879.diff --exclude='tests/\*'
git apply ../patches/pr27836.diff
cmake -B build -G Ninja -DGGML\_CUDA=ON -DCMAKE\_BUILD\_TYPE=Release
cmake --build build
```

### PR #27879 Details

Submitted by [tarruda](https://github.com/tarruda), assisted by GPT 5.6 Sol. Fixes:

1. **Sparse attention block selection** — QSA blocks formed in logical token order, preventing logit drift in long-context/multi-sequence inference
2. **Independent PLE embedding widths** — No longer assumes PLE KV projection equals main hidden width
3. **Metadata validation during loading** — Checks GDN, hyper-connection, QSA, compression, PLE invariants
4. **QSA indexer cache sequence copy** — Correctly copies indexer keys during sequence forks
5. **Recurrent-state rollback enabled** — GDN/PLE state can be correctly restored for speculative decoding
6. **Reject tensor-split mode** — Prevents unsupported modes from causing silent errors
7. **Metal alignment fix** — Dynamic threadgroup memory rounded to 16-byte alignment
8. **Metal index widening** — 16-bit → 32-bit, prevents overflow with large batches

### PR #27836 Details

Submitted by [rmonsurate](https://github.com/rmonsurate). Adds:

* NextN/MTP draft head support (`--spec-type draft-mtp`)
* Converter support (`--mtp` flag)
* Tested on M3 Max 128GB, UD-IQ4\_XS quantized output byte-identical with official

### License

Same as upstream [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp).

### AI Usage Disclosure

This project used AI-assisted tools during development:

* Mimo AI: Assisted in generating the initial README outline and provided `git` clone/branch-switch command examples.

All AI-generated suggestions and code snippets were reviewed, tested, and modified by humans. Final code is the responsibility of human contributors.

