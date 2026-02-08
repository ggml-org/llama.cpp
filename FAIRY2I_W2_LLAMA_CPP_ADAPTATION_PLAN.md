# Fairy2i-W2 在 llama.cpp 的适配规划（优先可在 Mac M4 16G 跑通）

> 工作目录：`/Users/liweitao/Downloads/Codefield/cpp/llama.cpp`  
> 模型目录：`models/Fairy2i-W2/`（HuggingFace 仓库镜像），论文链接见：`models/Fairy2i-W2/paper.md`  
> 参考现有实现：`gguf-py/convert_ifairy.py`、`src/llama-model.cpp` 中 `struct llm_build_ifairy`、以及 ifairy 的 ggml 量化/算子实现（`GGML_TYPE_IFAIRY`）

---

## 0. 目标、非目标、验收标准

### 0.1 目标（必须）

1. **能把 `models/Fairy2i-W2/` 转成 GGUF**（离线转换脚本）。
2. **llama.cpp 能加载该 GGUF 并用 `llama-cli` 正常生成**（输出非乱码、非全重复、非明显崩坏）。
3. **优先保证在 Mac M4 16G（CPU-only）可跑通**：显存/Metal 不是第一优先级。

### 0.2 非目标（第一阶段不做）

- 先不追求极致 tok/s；先保证正确性与可复现性。
- 先不做 Metal/BLAS 等后端的 IFAIRY 特化（大概率 CPU-only）。
- 先不做“把多阶段残差量化融合到单次 matmul”的内核级优化（可作为后续优化）。

### 0.3 验收标准（建议写死为 checklist）

- `cmake -B build-rel -DCMAKE_BUILD_TYPE=Release` && `cmake --build build-rel ...` 通过。
- 生成的 GGUF 满足：
  - `general.architecture` 为新架构（建议：`fairy2i`）。
  - 关键 KV（层数、头数、上下文长度、rope theta、rms eps）齐全。
  - 张量命名/形状与运行时代码一致（能 load）。
- `./build-rel/bin/llama-cli -m <fairy2i.gguf> ...` 能输出可读文本（见「7.3 Gibberish 检测」）。
- （强烈建议）与 PyTorch 参考实现做一个 **短 prompt、贪心解码** 的对齐/近似对齐验证（见「7.4 与 PyTorch 对比」）。

---

## 1. 模型现状与核心机制（适配前必须明确的假设）

### 1.1 Fairy2i-W2 是什么

从 `models/Fairy2i-W2/config.json` 可见它的 **外观仍是 Llama2-7B**：

- `model_type: "llama"`
- `hidden_size: 4096`
- `num_hidden_layers: 32`
- `num_attention_heads: 32`
- `intermediate_size: 11008`
- `max_position_embeddings: 4096`
- `rms_norm_eps: 1e-5`

HF repo 的 `load_model.py` 显示推理路径：

1) 先加载标准 `meta-llama/Llama-2-7b-hf` 架构  
2) 再加载 Fairy2i-W2 的权重（BF16，体积 ~13GB）  
3) 把所有 `nn.Linear` 替换成 `QATLinearComplexPhaseV2`（见 `models/Fairy2i-W2/qat_modules.py`）  

**结论**：权重文件里存的是“主权重 A（实矩阵）”，推理时会在 forward 把它转换/量化成等价的低比特形式。

### 1.2 Widely-linear 分解（必须完全复刻）

在 `models/Fairy2i-W2/qat_modules.py` 里，线性层权重 `A`（形状 `2n × 2m`）分块：

```
A = [[A11, A12],
     [A21, A22]]
```

并分解为两个复矩阵 `U, W`（各 `n × m`，复数）：

```
U_re = 0.5 * (A11 + A22)
U_im = 0.5 * (A21 - A12)
W_re = 0.5 * (A11 - A22)
W_im = 0.5 * (A12 + A21)
```

反向重构（同文件里用的等价关系）：

```
A11_q = W_re_q + U_re_q
A12_q = W_im_q - U_im_q
A21_q = W_im_q + U_im_q
A22_q = -W_re_q + U_re_q
```

### 1.3 量化：PhaseQuant + 2-stage residual（W2 的关键）

`models/Fairy2i-W2/quantization.py::PhaseQuantSTE` 与 `PhaseQuantSTE_V2` 给了明确算法：

- 码本：`{±1, ±i}`（轴向）
- scale：分别对“落在实轴象限”的元素取 `|w_real|` 的 **均值**，对“落在虚轴象限”的元素取 `|w_imag|` 的 **均值**（并 clamp 到 `>= 1e-6`）
- V2 residual：
  - stage0：对 `w` 做一次 PhaseQuant 得到 `q0`
  - stage1：对误差 `e = w - q0` 再做一次 PhaseQuant 得到 `q1`
  - 最终：`q = q0 + q1`

**注意**：为了能用 llama.cpp 里的 IFAIRY 2-bit 权重格式（每个元素只能表示 {±real_scale, ±i·imag_scale}），你需要把 **stage0 与 stage1 分开存储**（每个 stage 都是轴向码本），运行时做两次 matmul 再相加。

---

## 2. “复用 ifairy”是否足够？你现在方案里的关键遗漏点

你说“主要仿照 `convert_ifairy.py` 与 `llm_build_ifairy` 增加适配”，方向对，但**只改这两处通常跑不通**，主要遗漏在：

1. **架构差异**：Fairy2i-W2 的非线性/FFN/attention scaling 跟 iFairy 不同  
   - iFairy 在 `llm_build_ifairy` 里使用 `relu2`、`ggml_ifairy_mul`（复数乘法，语义为 `conj(a)*b`），并且 attention scaling 用 `1/sqrt(head_dim/2)`  
   - Fairy2i-W2 仍是 Llama2 的 **SiLU + elementwise mul（对实部/虚部分别相乘，无交叉项）**，并且 attention scaling 必须保持 **Llama 的 `1/sqrt(head_dim)`**（128）  
   → 不能直接复用 `llm_build_ifairy`，建议新建 `llm_build_fairy2i`

2. **线性层不是 “单个 ifairy 权重”**：每个 Linear 需要 `U/W × (stage0/stage1)`  
   → 每个线性层至少 4 个 IFAIRY 张量（`U0,U1,W0,W1`）。模型结构体/加载器/张量命名映射都要扩展。

3. **`ggml` 的 ifairy matmul 语义是 `w * conj(x)`**（见 `ggml_vec_dot_ifairy_q16_K_*` 注释）  
   - widely-linear 需要 `U*x + W*conj(x)`  
   - 现有内核天然给你 `W*conj(x)`（传入 `x` 即可）  
   - 但 `U*x` 需要你传入 `conj(x)`（因为内核会对输入取共轭）  
   → 需要实现/复用一个 **conjugate(input)** 的操作（最简单：split -> 翻转 imag -> merge；更优：新增 `ggml_ifairy_conj` unary op）。我觉得可以选`ggml_ifairy_conj` unary op

4. **GGUF/arch 接入点**：需要新增 `fairy2i` 架构映射与 KV 前缀  
   - `src/llama-arch.h/.cpp`：新增 `LLM_ARCH_FAIRY2I` 与字符串映射  
   - `gguf-py/gguf/constants.py`：新增 `MODEL_ARCH.FAIRY2I`（或至少能写 `general.architecture="fairy2i"`）  
   - `src/llama-model.cpp`：`load_hparams` / `load_tensors` / `build_graph` switch-case 增加新分支  

5. **转换脚本不能照搬 `convert_ifairy.py` 的输入假设**  
   - `convert_ifairy.py` 期望 HF 里有 `*_real/*_imag` 权重；Fairy2i-W2 的权重 key 是标准 Llama（`q_proj.weight` 等）  
   - Fairy2i 的 U/W 分解要从 **单个实矩阵 A** 推导出来  
   - W2 需要输出两阶段 residual 的 2-bit 权重  

---

## 3. 推荐的总体策略（优先正确 + 可在 M4 跑通）

### 3.1 推荐策略 A：新增 `fairy2i` 架构，线性层走 IFAIRY 2-bit，其他保持 Llama 语义

**核心思想**：

- 用 ggml 已有的 `GGML_TYPE_IFAIRY`（2-bit 复数轴向码本）来承载每个 stage 的 `U/W`。
- 图构建时维持 **ifairy 的“复数打包表示”**（每个 float 里塞两份 BF16：real/imag）。
- 在需要执行 Llama 的实数算子时（SiLU、elementwise mul、最终 logits），通过 `ggml_ifairy_split/merge` 在“打包复数 <-> 实数展开(2×)”之间切换。

优点：

- 内核/量化类型可复用 ifairy（CPU/NEON/LUT 路径有积累）。
- 权重体积接近论文目标（2 bits / real param）。

缺点：

- 每个 Linear 需要 4 次 matmul + 多次 add，**性能可能偏慢**（但先跑通）。

---

## 4. GGUF 设计（架构、KV、张量命名）

### 4.1 关键约定：在 GGUF 中存“复数维度”（half hidden)

为了让 ifairy 路径自然工作，建议：

- `n_embd`（GGUF `*.embedding_length`）存 **复数维度**：`4096/2 = 2048`
- `n_ff`（GGUF `*.feed_forward_length`）存 **复数维度（建议做 IFAIRY 对齐 padding）**：
  - 原始：`11008/2 = 5504`
  - 由于 `GGML_TYPE_IFAIRY` 的 block 约束 `k % QK_IFAIRY == 0`（`QK_IFAIRY=256`），而 `5504 % 256 != 0`
  - 推荐：`n_ff = GGML_PAD(5504, 256) = 5632`（对应 real 展开维度 `11264`，最后 256 个通道恒为 0，不影响原模型输出）
- `head_dim_real = (n_embd / n_head) * 2 = 128`（由 C++ hparams 推导）

这样运行时：

- “展开后的实向量长度”始终是 `2*n_embd = 4096`，与 Llama2 保持一致。

### 4.2 KV（建议最小集）

建议新增 `general.architecture = "fairy2i"` 后，写入：

- `fairy2i.context_length = 4096`
- `fairy2i.embedding_length = 2048`
- `fairy2i.block_count = 32`
- `fairy2i.feed_forward_length = 5632`（见上：为了满足 IFAIRY 的 `k % 256 == 0` 约束做 padding）
- `fairy2i.attention.head_count = 32`
- `fairy2i.attention.head_count_kv = 32`
- `fairy2i.attention.layer_norm_rms_epsilon = 1e-5`
- `fairy2i.rope.freq_base = 10000`
- `general.file_type = LLAMA_FTYPE_MOSTLY_IFAIRY`（复用 40）
- `fairy2i.vocab_size = 32000`

额外建议（为了未来支持 W1/W3 等）：

- `fairy2i.quant.residual_steps = 2`（uint32）
- `fairy2i.quant.codebook = \"{±1,±i}\"`（string，可选）

### 4.3 张量清单（建议命名）

下面以 layer `L` 表示第 `L` 层（0-based），`s` 表示 stage（0/1）。

**不量化或高精度**（建议与 iFairy 类似）：

- `token_embd`：F32（内容是打包 bf16 对的 float32），shape `(n_embd_complex=2048, vocab=32000)`
- `output_norm`：F32，shape `(2*n_embd_complex=4096)`
- `blk.L.attn_norm`：F32，shape `(4096)`（对应 HF `input_layernorm.weight`）
- `blk.L.ffn_norm`：F32，shape `(4096)`（对应 HF `post_attention_layernorm.weight`）
- `output`（lm_head）：F16，shape `(4096, 32000)`（即转置后的 [in, out]）

**线性层 2-bit（IFAIRY）**：

对每个线性层 `X ∈ {attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down}` 写：

- `blk.L.X.U.s0` / `blk.L.X.U.s1`（IFAIRY）
- `blk.L.X.W.s0` / `blk.L.X.W.s1`（IFAIRY）

其中每个张量 shape 都是 `(in_complex, out_complex)`（ggml 的“转置存储”习惯）。

---

## 5. 模型转换（gguf-py）详细规划

### 5.1 新脚本建议：`gguf-py/convert_fairy2i.py`

输入：

- `model_dir`：指向 `models/Fairy2i-W2/`
- `output_file`：输出 GGUF 路径
- 可选参数：`--residual-steps 2`、`--verbose`

输出：

- `fairy2i-w2.gguf`（建议放 `models/Fairy2i-W2/` 下或 `models/` 下）

### 5.2 转换流程（按权重类型分流）

#### 5.2.1 词表/Tokenizer

复用 `convert_ifairy.py` 的 `set_vocab()` 逻辑：

- `gguf.LlamaHfVocab(Path(model_dir))`
- `gguf.SpecialVocab(...).add_to_gguf(writer)`

#### 5.2.2 Embedding：`model.embed_tokens.weight`

HF：shape `(vocab, hidden=4096)`，BF16  
目标：`token_embd` shape `(n_embd_complex=2048, vocab)` 的“打包复数”F32

打包规则（必须与 ggml ifairy split/merge 一致）：

- 对每个 token `t`、每个 complex idx `i`：
  - real = W[t, i]
  - imag = W[t, i + n_embd_complex]
  - pack 到一个 32-bit word：低 16 bits = real(bf16)，高 16 bits = imag(bf16)

#### 5.2.3 RMSNorm：`input_layernorm/post_attention_layernorm/model.norm`

HF：shape `(4096,)`，BF16  
目标：F32，shape `(4096,)`

#### 5.2.4 Linear（核心）：把 A 转成 U/W 并做 2-stage PhaseQuant

适用 key：

- `model.layers.L.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.L.mlp.{gate,up,down}_proj.weight`

步骤（单个线性层）：

1) 读取 BF16 的 `A`（shape `out × in`，均为偶数）  
2) 分块 `A11,A12,A21,A22`（输出/输入维度各一分为二）  
3) 计算 `U_re,U_im,W_re,W_im`  
4) 对 `U` 做 PhaseQuant V2，得到 `U_stage0`、`U_stage1`（两个 stage 都是轴向）  
5) 对 `W` 同理  
6) 把每个 stage 写入 `GGML_TYPE_IFAIRY`（GGUF quant type `F16_I2`），命名按「4.3」

##### 关键细节 A：PhaseQuant 的分类不必用 angle（但要保持 tie 规则）

PhaseQuant 的 45° 分界等价于：

- 若 `abs(real) > abs(imag)` 则选 real 轴，否则选 imag 轴（tie 归 imag）

这比对整矩阵做 `angle()` 更省时、更省内存。

##### 关键细节 B：IFAIRY 权重的 scale（极容易踩坑）

ggml 的 ifairy `vec_dot` 实现里，权重 scale **通常从每行的第一个 block 读取**（`w[0].d_real/d_imag`）。因此转换器必须保证：

- 每行的 `block0` 一定写入该行正确的 `(d_real, d_imag)`（即该 stage 的 `s_re/s_im`），即使该 block0 恰好没有某轴的权重 code 也没关系，**scale 不能为 0**。

最简单且安全的做法：

- 对每个 stage，计算出 `s_re/s_im`（标量），然后把它写进该行所有 blocks（至少写进 block0）。

建议直接实现一个 IFAIRY packer（参考 `ggml/src/ggml-quants.c::quantize_row_ifairy_ref` 的布局），**不要完全依赖** `gguf-py` 的 per-block scale 行为。

##### 关键细节 C：权重 codes 的 byte packing/layout 必须匹配 ggml 解码

ggml 解码逻辑（见 `ggml_vec_dot_ifairy_q16_K_generic`）使用 `chunk/lane/part` 的布局：

> `|0 16 32 48|1 17 33 49|...|15 31 47 63|`

因此 packer 要么：

- 直接按最终布局 pack（推荐），要么  
- 先按顺序 pack 再调用 `convert_ifairy.py` 里的 `repack_ifairy_blocks()`

---

## 6. llama.cpp 侧适配规划（C++ 需要改哪些点）

### 6.1 新增架构枚举与 GGUF KV 前缀

需要修改：

- `src/llama-arch.h`：新增 `LLM_ARCH_FAIRY2I`
- `src/llama-arch.cpp`：
  - arch name mapping 增加 `"fairy2i"`
  - KV key 格式串（`"%s.embedding_length"` 等）要支持 `fairy2i.*`
  - Tensor name map：为 Fairy2i 增加张量名模板（至少包含 norms、output、以及所有 U/W stage 名）

### 6.2 模型结构体：每个 Linear 需要 4 个张量（U/W × 2 stage）

推荐结构：

```cpp
struct llama_widely_linear_ifairy {
    std::array<ggml_tensor *, 2> U; // stage0, stage1
    std::array<ggml_tensor *, 2> W; // stage0, stage1
};
```

然后在 `struct llama_layer` 里新增（只列 Llama2 需要的）：

- `llama_widely_linear_ifairy wq, wk, wv, wo;`
- `llama_widely_linear_ifairy ffn_gate, ffn_up, ffn_down;`

### 6.3 加载器：按命名把 4×weights 填进去

需要在 `src/llama-model.cpp` 的张量加载逻辑里新增 Fairy2i 分支：

- `blk.L.attn_q.U.s0` 等 → `layers[L].wq.U[0]` …
- 类似地加载 K/V/O、FFN 三个线性层

并做 shape/类型断言：

- 所有 IFAIRY 权重张量 type = `GGML_TYPE_IFAIRY`
- norms 长度为 `2*n_embd`（这里是 4096）

### 6.4 hparams：注意 head_dim 的推导与 eps key

对 Fairy2i，建议像 ifairy 一样把：

- `n_embd_head_k/v = (n_embd_complex / n_head) * 2`

这样：

- `head_dim_real` = 128（与 Llama2 一致）
- 但 Q/K/V 在 packed 复数表示下每头长度是 `head_dim_real/2` = 64（与 reshape 逻辑一致）

同时 eps 用 RMS key：

- 读取 `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` → `hparams.f_norm_rms_eps`

### 6.5 图构建：新建 `llm_build_fairy2i`（不要直接改 ifairy）

核心原因：FFN 非线性与 attention scaling 不同。

#### 6.5.1 必须新增的 helper：widely-linear matmul

建议在 `llm_graph_context` 加一个 helper（伪代码）：

```
wide_linear(x, x_conj, U[2], W[2]):
    yU = mul_mat(U[0], x_conj) + mul_mat(U[1], x_conj)   // 因为 mul_mat 内部会 conj(input)
    yW = mul_mat(W[0], x)      + mul_mat(W[1], x)
    return yU + yW
```

这里 `mul_mat(W, x)` 实际是 `W * conj(x)`（ggml ifairy 语义）。

#### 6.5.2 conj(x) 的实现（先正确，后优化）

最稳的实现（不改 ggml）：

1) `x_split = ggml_ifairy_split(x)` → float32 长度 `2*n`  
2) 乘一个 sign 向量 `[+1...+1, -1...-1]` 把 imag 半区翻转  
3) `x_conj = ggml_ifairy_merge(x_split_flipped)`

后续优化（可选）：

- 新增 `GGML_OP_IFAIRY_CONJ`（直接翻转每个 packed float 的 imag BF16 符号位），避免 split/merge。

#### 6.5.3 Attention：**scaling 必须对齐 Llama2（这是常见遗漏点）**

Fairy2i-W2 基于 Llama2，因此 attention scaling 应为：

- `kq_scale = 1.0 / sqrt(head_dim_real)`（即 `1/sqrt(128)`）

而 `llm_build_ifairy` 目前用的是 `1/sqrt(head_dim_real/2)`，直接照抄会让行为偏离 HF 参考实现。

#### 6.5.4 Attention 计算图（建议走 ifairy 的 split/rope/mha/merge 框架）

单层大致流程（保持 Llama 语义）：

1) `inpL`（packed complex, len=2048）
2) `cur = norm(inpL)`（RMSNorm，scale 向量长度 4096）
3) 计算 `cur_conj`（packed）
4) `Q = wide_linear(cur, cur_conj, wq)` → packed
5) `K = wide_linear(cur, cur_conj, wk)` → packed
6) `V = wide_linear(cur, cur_conj, wv)` → packed
7) reshape 到 `(head_dim_real/2, n_head, n_tokens)` 后：
   - `Q = ggml_ifairy_rope(Q, pos, n_rot=head_dim_real, ...)` → split float32（len=head_dim_real）
   - `K = ggml_ifairy_rope(K, pos, ...)` → split float32
   - `V = ggml_ifairy_split(V)` → split float32
8) 用现有 `build_attn_mha()` 做 attention（softmax 等在实数域）
9) `attn_out = ggml_ifairy_merge(attn_out_split)` → packed
10) `proj = wide_linear(attn_out, conj(attn_out), wo)` → packed
11) `inpL = ggml_ifairy_add(proj, inpL)`（残差加）

#### 6.5.5 FFN（Llama2 的 SiLU-gated MLP）

关键点：Llama 的 gated MLP 是“实数逐元素乘”，不是复数乘法；不能用 `ggml_ifairy_mul`。

流程建议：

1) `cur = norm(inpL)`（packed）
2) `cur_conj = conj(cur)`（packed）
3) `gate = wide_linear(cur, cur_conj, ffn_gate)` → packed（complex dim = 5632，其中最后 128 个 complex/256 个 real 通道为 0）
4) `up   = wide_linear(cur, cur_conj, ffn_up)`   → packed（complex dim = 5632）
5) `gate_s = ggml_ifairy_split(gate)` → float32 len `2*5632 = 11264`
6) `up_s   = ggml_ifairy_split(up)`   → float32 len 11264
7) `gate_s = ggml_silu(gate_s)`（实数 SiLU）
8) `mul_s  = ggml_mul(gate_s, up_s)`（实数逐元素）
9) `mul    = ggml_ifairy_merge(mul_s)` → packed
10) `down  = wide_linear(mul, conj(mul), ffn_down)` → packed（回到 2048 complex；ffn_down 的输入 dim 之所以能用 IFAIRY，是因为我们把 n_ff pad 到 5632）
11) `inpL  = ggml_ifairy_add(down, inpL)`

#### 6.5.6 输出 logits

1) `cur = norm(inpL)`（packed）
2) `cur_s = ggml_ifairy_split(cur)` → float32 len 4096
3) `logits = ggml_mul_mat(output, cur_s)`（output 为 F16，shape `(4096, vocab)`）

---

## 7. 测试与验证（包含“是否 gibberish”）

### 7.1 无权重的单元测试（强烈建议先做，能提前抓 80% 错误）

目标：在不依赖 7B 权重的情况下，验证三件事：

1) A→U/W 的分解公式正确  
2) PhaseQuant V2 的 stage0/stage1 与 PyTorch 参考一致  
3) IFAIRY pack/layout 与 ggml 解码一致

建议做法：

- 写一个小维度（例如 in/out=64 或 128，保证可被 2 整除且能被 `QK_IFAIRY=256` 限制绕开/或使用更小的自定义 block 仅用于测试）  
- 或者在测试里直接复用 ggml 的 `quantize_row_ifairy_ref`（它能把 real/imag float32 量化成 block_ifairy）来做 end-to-end 对照。

### 7.2 GGUF 结构自检（转换后立刻做）

- 用 `./build-rel/bin/llama-gguf <file> r n` 检查：
  - KV keys 是否齐
  - tensor 名称是否全部存在
  - tensor 形状是否符合预期

### 7.3 `llama-cli` 的 Gibberish 检测（你要求的项）

建议用 **确定性设置（贪心）**，避免采样噪声掩盖错误：

```bash
./build-rel/bin/llama-cli \
  -m /path/to/fairy2i-w2.gguf \
  -p "I believe life is" \
  -n 64 \
  --seed 1 \
  --temp 0 \
  --top-k 1 \
  --threads 8 \
  -ngl 0 \
  -c 2048 \
  -no-cnv
```

判定建议（主观 + 简单客观）：

- 主观：输出是否像自然语言，而非大量乱码、重复 token、奇怪符号。
- 客观（可选）：统计输出里不可打印字符比例、连续重复 token 段长度等。

### 7.4 与 PyTorch 参考对比（强烈建议）

用 `models/Fairy2i-W2/load_model.py` 跑同样 prompt，设置 `do_sample=False`（贪心）并固定 `max_new_tokens`，对比：

- 前 N 个 token 是否大体一致（完全一致不一定要求，但差距过大通常表示某个环节错了：packing/transpose/kq_scale/rope/norm）。

### 7.5 性能/资源（面向 M4）

在能正确生成后再考虑：

- `GGML_IFAIRY_LUT=1`（如果你使用了带 LUT 的构建，如 `build-rel-lut`）  
- `GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1`（让 matmul 对 activation 的量化策略更可控，可能影响性能/内存）

---

## 8. 在 Mac M4 16G 上的推荐运行方式（从“能跑”开始）

### 8.1 构建

按 repo 指南：

```bash
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel -j $(sysctl -n hw.ncpu)
```

如果要试 LUT（可选）：

- 参考现有 `build-rel-lut/` 的构建方式（通常带 `GGML_IFAIRY_LUT_CPU`）。

### 8.2 转换

建议输出到 `models/Fairy2i-W2/fairy2i-w2.gguf`：

```bash
python3 gguf-py/convert_fairy2i.py models/Fairy2i-W2 models/Fairy2i-W2/fairy2i-w2.gguf --residual-steps 2
```

### 8.3 运行（CPU-only）

```bash
./build-rel/bin/llama-cli \
  -m models/Fairy2i-W2/fairy2i-w2.gguf \
  -p "Hello, how are you?" \
  -n 64 \
  --seed 1 \
  --temp 0 \
  --top-k 1 \
  --threads 8 \
  -ngl 0 \
  -c 2048 \
  -no-cnv
```

---

## 9. 常见错误与定位清单（尤其针对 gibberish）

按出现频率从高到低：

1. **token_embd 打包顺序错**（real/imag 半区反了，或 bf16 half 放反了）  
   - 现象：几乎必 gibberish
   - 排查：拿同一 token 的 embedding，split 后检查前 2048/后 2048 是否与 HF 对齐

2. **权重 transpose 方向错**（ggml 的 `mul_mat` 约定是“权重转置存储”）  
   - 现象：输出完全崩坏
   - 排查：确认所有 IFAIRY 张量 shape 是 `(in_complex, out_complex)` 而不是反过来

3. **attention scaling 用错**（把 `1/sqrt(128)` 写成 `1/sqrt(64)`）  
   - 现象：输出质量显著下降，可能出现重复/跑偏
   - 排查：对齐 HF Llama2 scaling

4. **conj(x) 实现错**（imag 符号翻转没做/做反/翻转了 real）  
   - 现象：线性层输出失真严重

5. **IFAIRY 权重 scale 读取问题**（某行 block0 scale=0）  
   - 现象：部分行输出异常，整体质量崩
   - 解决：转换器按行写 `d_real/d_imag`，至少保证 block0 正确

6. **FFN 用了复数乘法**（误用了 `ggml_ifairy_mul`）  
   - 现象：输出偏离 HF，可能很明显
   - 解决：split 后用 `ggml_mul` 做实数逐元素乘

7. **忽略 IFAIRY 的 `k % 256 == 0` 约束**（典型踩点：Llama2 的 down_proj 输入维度 11008 real / 5504 complex）  
   - 现象：转换时 assert / 运行时 matmul 直接报错或 silent wrong  
   - 解决：把 `n_ff` pad 到 `5632 complex / 11264 real`，并在转换时把 gate/up/down 的 A 矩阵按行/列补 0

---

## 10. 后续优化方向（正确性通过后再做）

1. **减少 matmul 次数**：把 `stage0+stage1` 融合到一次 kernel（需要改 ggml 内核/权重格式）。
2. **新增 `ggml_ifairy_conj`**：避免 split/merge 做共轭，减少图节点与内存带宽。
3. **输出层量化**：可选，把 `lm_head` 也纳入 Fairy2i 量化（但要确认论文/作者是否这么做，且要评估质量与性能）。
4. **更严格的回归测试**：固定 seed + 贪心输出的 token 序列 hash，作为 CI gate（可选）。
