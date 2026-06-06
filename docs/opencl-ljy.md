# [2026-06-04] iFairy OpenCL算子对齐CPU实现与回归测试接入

## 本次工作概述（可用于PR描述）

本次工作围绕 iFairy 在 OpenCL 后端的功能补齐与可回归验证展开，目标是让 OpenCL 结果与 CPU 版本行为一致，并补齐关键路径算子支持。

主要完成内容：

- 新增并接入 OpenCL 算子：split、merge、relu2。
- 新增 iFairy rope 的 OpenCL 实现（参考 CPU 版本逻辑）。
- 在 OpenCL 后端完成对应算子的 capability 判定与 compute_forward 分发接线。
- 新增最小回归用例：对比 CPU 与 OpenCL 在 split/merge/relu2 上的一致性。
- 在 Windows + shared 构建链路下，修复并收敛一批跨模块符号链接问题（导出/可见性/C与C++符号形式相关）。

## 主要代码变更点

- OpenCL kernel 与构建清单

  - ggml/src/ggml-opencl/kernels/ifairy_split.cl
  - ggml/src/ggml-opencl/kernels/ifairy_merge.cl
  - ggml/src/ggml-opencl/kernels/ifairy_relu2.cl
  - ggml/src/ggml-opencl/kernels/rope.cl（新增 iFairy rope kernel）
  - ggml/src/ggml-opencl/CMakeLists.txt（新增 kernel 纳入）
- OpenCL后端调度与支持判定

  - ggml/src/ggml-opencl/ggml-opencl.cpp
    - 新增 split/merge/relu2/ifairy rope 的 kernel 加载、supports_op 判定与 compute_forward 分发。
- 回归测试

  - tests/test-ifairy.cpp
    - 新增最小回归测试入口（CPU/OpenCL 一致性对比）。
    - 新增命令行参数：--ifairy-opencl-regression。
- 链接可见性修复（Windows shared）

  - ggml/src/ggml-ifairy-lut.cpp
  - ggml/src/ggml-ifairy-lut.h
  - ggml/src/ggml-cpu/ggml-cpu.c

## 当前状态

- OpenCL后端新增算子代码已接入并可编译。
- split、merge、relu2、ifairy rope 均已完成 OpenCL 接线。
- 回归测试代码已落地，Windows shared 下 test-ifairy 的完整链接链路仍在持续收敛中（属于构建/导出链路问题，不是算子逻辑问题）。

## 备注

- 本次实现优先保证：
  1) 算子语义与CPU实现对齐；
  2) 最小可回归验证路径可用；
  3) OpenCL后端增量改动清晰、可审查。
