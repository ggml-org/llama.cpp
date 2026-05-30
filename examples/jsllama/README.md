# jsllama

A lightweight Bun.js FFI binding for llama.cpp, enabling direct inference calls to shared libraries (.so) for running large language models in JavaScript.

## Features

- Load GGUF format large language models
- Streaming text generation (generator and callback styles)
- Model unloading and resource cleanup
- Supports CPU and GPU (Vulkan) inference
- Temperature sampling and repetition penalty to avoid duplicate output
- Zero native compilation needed at runtime — pre-compiled shared libraries

## Prerequisites

- [Bun.js](https://bun.sh/) >= 1.0
- llama.cpp compiled shared library files (included in `bin/` directory)

## Installation

```bash
cd jsllama
bun install
```

## Compilation (Optional)

If you need to recompile the C wrapper library, use the following command:

```bash
cd <PROJECT_DIR>
g++ -shared -fPIC -o bin/libjsllama.so jsllama.cpp \
  -I<LLAMA_CPP_INCLUDE_DIR> \
  -I<GGML_INCLUDE_DIR> \
  -L./bin -lllama \
  -Wl,-rpath,'$ORIGIN'
```

Replace the placeholders:
- `<PROJECT_DIR>` — Path to the jsllama project directory
- `<LLAMA_CPP_INCLUDE_DIR>` — Path to llama.cpp `include/` directory
- `<GGML_INCLUDE_DIR>` — Path to ggml `include/` directory

## Usage

### Basic Usage

```javascript
import { Llama } from "./index.js";

const llama = new Llama({
  nCtx: 2048,
  nBatch: 512,
  nThreads: 4,
  nGpuLayers: -1,
});

llama.loadModel("/path/to/model.gguf");

for (const token of llama.generate("Hello, introduce yourself", { maxTokens: 100 })) {
  process.stdout.write(token);
}

llama.unloadModel();
Llama.free();
```

### Streaming Output (with Callback)

```javascript
await llama.generateStream("Tell me a short joke: ", (token) => {
  process.stdout.write(token);
}, { maxTokens: 50 });
```

### Configuration Options

```javascript
const llama = new Llama({
  nCtx: 2048,
  nBatch: 512,
  nThreads: 4,
  nGpuLayers: -1,
});
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| nCtx | number | 2048 | Context length for text processing |
| nBatch | number | 2048 | Batch size for processing |
| nThreads | number | 4 | Number of threads during generation |
| nGpuLayers | number | -1 | Layers to offload to GPU (-1 = all) |

### Generation Options

```javascript
llama.generate(prompt, {
  maxTokens: 256,
});
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| maxTokens | number | 256 | Maximum number of tokens to generate |

## Running Examples

```bash
bun example.js /path/to/your/model.gguf
```

## Running Tests

```bash
bun test.js
```

## Project Structure

```
jsllama/
├── bin/
│   ├── libjsllama.so      # C wrapper library
│   ├── libllama.so*       # llama.cpp core library
│   ├── libggml*.so*       # ggml base library
│   └── libllama-common.so* # llama.cpp common library
├── jsllama.cpp            # C wrapper source code
├── dll.js                 # FFI binding layer
├── index.js               # High-level JavaScript API
├── example.js             # Usage example
├── test.js                # Test script
└── package.json
```

## Architecture

1. **jsllama.cpp** — C wrapper layer that simplifies the complex llama.cpp API
   - Encapsulates `llama_batch` structure with automatic `n_tokens` setup
   - Configures temperature sampler and repetition penalty to avoid duplicate output
2. **dll.js** — Bun.js FFI bindings that define C function signatures
3. **index.js** — High-level JavaScript class providing an easy-to-use API
   - `generate()` — Generator style, iterate with `for...of`
   - `generateStream()` — Callback style, async streaming output

---

# jsllama

基于 Bun.js FFI 的 llama.cpp 封装库，支持直接调用 .so 文件进行模型推理。

## 功能

- 加载 GGUF 格式的大语言模型
- 流式文本生成输出（支持生成器和回调两种方式）
- 模型卸载与资源释放
- 支持 CPU 和 GPU (Vulkan) 推理
- 温度采样和重复惩罚，避免生成重复内容

## 前置要求

- [Bun.js](https://bun.sh/) >= 1.0
- llama.cpp 编译的共享库文件（已包含在 `bin/` 目录中）

## 安装

```bash
cd jsllama
bun install
```

## 编译（可选）

如需重新编译 C 封装库，请使用以下命令：

```bash
cd <项目目录>
g++ -shared -fPIC -o bin/libjsllama.so jsllama.cpp \
  -I<LLAMA_CPP_INCLUDE_DIR> \
  -I<GGML_INCLUDE_DIR> \
  -L./bin -lllama \
  -Wl,-rpath,'$ORIGIN'
```

占位符说明：
- `<项目目录>` — jsllama 项目目录路径
- `<LLAMA_CPP_INCLUDE_DIR>` — llama.cpp 的 `include/` 目录路径
- `<GGML_INCLUDE_DIR>` — ggml 的 `include/` 目录路径

## 使用方法

### 基本用法

```javascript
import { Llama } from "./index.js";

const llama = new Llama({
  nCtx: 2048,
  nBatch: 512,
  nThreads: 4,
  nGpuLayers: -1,
});

llama.loadModel("/path/to/model.gguf");

for (const token of llama.generate("你好，请介绍一下自己", { maxTokens: 100 })) {
  process.stdout.write(token);
}

llama.unloadModel();
Llama.free();
```

### 流式输出（带回调）

```javascript
await llama.generateStream("讲一个简短的笑话：", (token) => {
  process.stdout.write(token);
}, { maxTokens: 50 });
```

### 配置选项

```javascript
const llama = new Llama({
  nCtx: 2048,
  nBatch: 512,
  nThreads: 4,
  nGpuLayers: -1,
});
```

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| nCtx | number | 2048 | 文本上下文长度 |
| nBatch | number | 2048 | 批处理大小 |
| nThreads | number | 4 | 生成时使用的线程数 |
| nGpuLayers | number | -1 | 卸载到 GPU 的层数（-1 表示全部） |

### 生成选项

```javascript
llama.generate(prompt, {
  maxTokens: 256,
});
```

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| maxTokens | number | 256 | 最大生成 token 数 |

## 运行示例

```bash
bun example.js /path/to/your/model.gguf
```

## 运行测试

```bash
bun test.js
```

## 项目结构

```
jsllama/
├── bin/
│   ├── libjsllama.so      # C 封装库
│   ├── libllama.so*       # llama.cpp 核心库
│   ├── libggml*.so*       # ggml 基础库
│   └── libllama-common.so* # llama.cpp 公共库
├── jsllama.cpp            # C 封装源码
├── dll.js                 # FFI 绑定层
├── index.js               # 高级 JavaScript API
├── example.js             # 使用示例
├── test.js                # 测试脚本
└── package.json
```

## 架构说明

1. **jsllama.cpp** — C 封装层，简化 llama.cpp 的复杂 API
   - 封装 `llama_batch` 结构体，自动设置 `n_tokens`
   - 配置温度采样器和重复惩罚，避免重复输出
2. **dll.js** — Bun.js FFI 绑定，定义 C 函数签名
3. **index.js** — 高级 JavaScript 类，提供易用的 API
   - `generate()` — 生成器方式，使用 `for...of` 遍历
   - `generateStream()` — 回调方式，异步流式输出
