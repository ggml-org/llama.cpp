# IFAIRY64 — 状态 / 性能记录

Status: Draft (2026-04-23)

本文件用于记录 `IFAIRY64` 相关的实现状态、性能数据和专项变更，避免继续混在 `IFAIRY_ARM_3W_LUT_V2_STATUS.md` 中。

相关文档：
- `IFAIRY64_LUT_IMPLEMENTATION_PLAN.md`
- `IFAIRY64_X86_ADAPTATION_EXECUTION_GUIDE.md`
- `IFAIRY_ARM_3W_LUT_V2_STATUS.md`（旧的 ARM 3W LUT V2 总状态）

---

## 变更记录（Changelog）

按日期追加（YYYY-MM-DD）：

### 2026-04-22 (working tree; base build `abcaafef`)
- 变更摘要：
  - `IFAIRY64` 在 `GGML_IFAIRY_LUT=1` 时于模型加载阶段提前完成 LUT transform/prepack，避免 decode 首轮再做 transform。
  - `IFAIRY64` 的 packed weight tile 不再把每 lane scale 展开成 `f32`，改为保留 `fp16` scale 并在 kernel 内按需转 `f32`，将 packed footprint 从 `384B/block-tile` 压回 `320B/block-tile`。
- Correctness:
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
- microbench（Machine: Mac16,12 / Apple M4）：
  - `./build-rel-lut/bin/ifairy-microbench --type ifairy64 --mode fused --m 3456 --k 2560 --iters 200 --warmup 20`
  - Result:
    - `ns/iter=322370.0`
- `llama-bench`（model=`~/fairy2i_32b/fairy2i_32b.gguf`; threads=4; `-dev none -ngl 0 -b 32 -ub 32 -fa 0 --no-warmup -p 0 -n 32 -r 1 -o md`）：
  - vecdot baseline:
    - `./build-rel-lut/bin/llama-bench -m ~/fairy2i_32b/fairy2i_32b.gguf ...`
    - `tg32`: `2.22 ± 0.00 tok/s`
  - explicit LUT:
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench -m ~/fairy2i_32b/fairy2i_32b.gguf ...`
    - `tg32`: `2.46 ± 0.00 tok/s`
  - Delta vs vecdot:
    - `tg32`: `+10.8%`

### 2026-04-23 (working tree, Fairy2i 32B output-only merged IFAIRY64)
- 变更：
  - 新增 `LLAMA_FAIRY2I_MERGED_OUTPUT=1` opt-in。
  - load 阶段从 `output.{U,W}.s{0,1}` 逐行 `dequant + sum + requant` 合成两块 output-only `IFAIRY64` 权重：
    - `output.U.merged`
    - `output.W.merged`
  - 最终 lmhead 从 4 次 Fairy2i output matmul 缩成 2 次；中间层不变。
- 额外权重开销：
  - `load_tensors: prepared merged FAIRY2I output weights in 0.523 sec (2 x 58.01 MiB)`
- 验证：
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
  - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 ./build-rel-lut/bin/llama-cli -m ~/fairy2i_32b/fairy2i_32b.gguf -dev none -ngl 0 -t 4 -c 8196 -b 32 -ub 32 -fa off --no-warmup -no-cnv --temp 0.2 --top-k 20 --top-p 0.9 -n 1 -p '<｜begin▁of▁sentence｜> You are a helpful AI assistant. <｜User｜> Where is China?'`: smoke PASS
- `llama-bench`（model=`~/fairy2i_32b/fairy2i_32b.gguf`, threads=4, `-p 0 -n 32 -r 1`）：
  - 默认 vecdot：
    - `./build-rel-lut/bin/llama-bench ...`: `tg32=2.24 tok/s`
    - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.26 tok/s`
    - 提升：`+0.02 tok/s`（`+0.9%`）
  - 显式 LUT：
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.42 tok/s`
    - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.49 tok/s`
    - 提升：`+0.07 tok/s`（`+2.9%`）
- 结论：
  - 这条 output-only 合并对 LUT 和 vecdot 都是正收益，但收益量级符合 lmhead 占比，属于小幅提速。
  - 当前只做了 smoke 和 tok/s 验证，未做质量/困惑度回归。
