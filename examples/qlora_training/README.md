# llama.cpp — Native QLoRA Training

Native QLoRA + Reward-Weighted SFT training pipeline for quantized GGUF models.

The base model weights remain **frozen** (quantized tensors are skipped by `llama_set_param` because they are not `GGML_TYPE_F32`). Only freshly-allocated F32 LoRA A/B tensors are trained. The saved adapter GGUF is directly compatible with the existing `llama_adapter_lora_init` loader and `llama-export-lora` merge tool.

**Status:** Working. Phase 1 (QLoRA SFT) and Phase 2 (Reward-Weighted SFT) are implemented and functional. Training speed is currently limited by full backprop through quantized weights — see [Known Limitations](#known-limitations).

---

## Build

```bash
cd /mnt/w/llm-trading-arena/unsloth-api/llama.cpp

# First time (CUDA build):
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build -j$(nproc)

# Incremental rebuild (after code changes):
cmake --build build --target llama-finetune-qlora -j$(nproc)
# If llama-adapter.cpp or llama-context.cpp changed, rebuild all:
cmake --build build -j$(nproc)

# ROCm build:
./ROCm-build.sh

# Optional: compile for a specific AMD GPU architecture:
GPU_TARGETS=gfx1100 ./ROCm-build.sh

# Radeon 680M (gfx1035, unsupported test configuration):
# ROCm's packaged rocBLAS kernels target gfx1030, so use a separate build.
BUILD_DIR=build-rocm-gfx1030 GPU_TARGETS=gfx1030 ./ROCm-build.sh

# Incremental ROCm rebuild:
cmake --build build-rocm --target llama-finetune-qlora -j$(nproc)
```

Run the Radeon 680M compatibility build with:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
GGML_CUDA_DISABLE_GRAPHS=1 \
./build-rocm-gfx1030/bin/llama-finetune-qlora ...
```

The Radeon 680M is not in AMD's supported ROCm GPU matrix. This compatibility
mode is intended for local testing and uses the packaged gfx1030 rocBLAS
kernels. Do not combine gfx1030 and gfx1035 objects in one build directory.

The ROCm build supports the same SFT, reward-weighted SFT, GRPO, resume,
checkpointing, LoRA QAT, partial offload, and optimizer modes as the CUDA build.
HIP compiles the shared `ggml-cuda` training kernels, including quantized
`OUT_PROD`, `OUT_PROD_ID`, and device-resident Q8 AdamW state.

---

## Phase 1 — QLoRA SFT (`llama-finetune-qlora`)

Trains LoRA adapters on a quantized GGUF model.

### Recommended command (1.7B model, 16 GB card)

```bash
./build/bin/llama-finetune-qlora \
  --model ~/qwen3-1.7b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --lora-rank 16 --lora-alpha 16 \
  -c 4096 -b 4096 -ub 512 \
  --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 -lr-min 1e-6 \
  --save-every 10 \
  --lora-out ~/adapter.gguf \
  --epochs 3 --seed 42
```

### Recommended command (15B model, 16 GB card, partial offload)

```bash
./build/bin/llama-finetune-qlora \
  --model ~/nemotron-15b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --lora-rank 16 --lora-alpha 16 \
  -ngl 13 -c 14336 -b 14336 -ub 1024 \
  --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 -lr-min 1e-6 \
  --save-every 8 \
  --lora-out ~/nemotron-lora.gguf \
  --epochs 3 --seed 42
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to quantized GGUF model |
| `--train-file` | *(required)* | JSONL training dataset |
| `-dthr` / `--dataset-threads` | physical CPU cores | JSONL parse, template, and tokenization worker threads |
| `--lora-rank` | `16` | LoRA rank r |
| `--lora-alpha` | `0` (= rank) | LoRA alpha; effective scale = alpha/rank |
| `--lora-targets` | see below | Comma-separated internal tensor name substrings |
| `--lora-out` | `adapter.gguf` | Output adapter GGUF path (supports `~`) |
| `--resume` | *(none)* | Resume weights and training position from a `--save-every` checkpoint |
| `--save-every` | `0` | Save checkpoint every N dataset windows (0 = end only) |
| `--freeze-layers` | `0` | Skip LoRA on first N transformer layers (blk.0..N-1); backward already pruned automatically |
| `--grad-checkpoint` | `0` | Mark every Nth forward node persistent to reduce activation VRAM; good values: 32–64 |
| `--train-on-prompt` | off | Compute loss on prompt tokens too (default: response-only loss) |
| `--shuffle-dataset` | off | Shuffle dataset windows at the start of each epoch |
| `--val-split` | `0.0` | Fraction of data to hold out for validation (e.g. `0.1` = 10%); val loss logged per epoch |
| `-epochs` / `--epochs` | `3` | Training epochs |
| `-c` / `--ctx-size` | `512` | Training context window (tokens) |
| `-b` / `--batch-size` | `2048` | Tokens per `llama_decode` call; set equal to `-c` |
| `-ub` / `--ubatch-size` | `512` | GPU micro-batch tokens; controls VRAM vs. step time |
| `-ngl` | `999` | GPU layers to offload |
| `-lr` / `--learning-rate` | `1e-4` | AdamW learning rate |
| `--lr-scheduler` | `constant` | Learning-rate schedule: `constant` or `cosine` |
| `--warmup-steps` | `0` | Linear warmup over N logical training steps |
| `--warmup-init-ratio` | `0.1` | Initial warmup LR as a fraction of peak LR; must be in `(0, 1]` |
| `--verbose-loss` | off | Print one structured loss line per SFT dataset window |
| `-lr-min` / `--learning-rate-min` | `-1` | Cosine floor; values below 0 use 0 |
| `--seed` | `42` | Random seed for LoRA init |

For SFT, one scheduler step is one completed dataset window. For GRPO, it is
one completed GRPO iteration. The recommended `-b == -c` configuration performs
one optimizer update per SFT scheduler step. Micro-batches used for gradient
accumulation do not advance the schedule. Checkpoints store the completed
scheduler step, so warmup and cosine decay continue from the same position after
resume. Use the same scheduler, warmup steps, warmup init ratio, base learning
rate, and minimum learning rate flags when resuming.

During warmup, step 0 starts at `peak_lr * warmup_init_ratio`. Each subsequent
logical step linearly interpolates toward peak LR using
`progress = step / warmup_steps`. The ratio is ignored when warmup is disabled.

With `--verbose-loss`, each completed SFT window prints one line:

```text
epoch=1 window=42 window_loss=1.234567 ema16=1.345678 ema64=1.456789 epoch_mean=1.567890 lr=1e-05
```

`window_loss` is the mean of the ubatch losses in that window. EMA16 uses
`0.8825 * ema16 + 0.1175 * window_loss`; EMA64 uses
`0.9692 * ema64 + 0.0308 * window_loss`. The EMAs continue across epochs,
while `epoch_mean` resets at each epoch.

### Resume from a checkpoint

Use the same model, dataset, context size, and validation split as the original run. `--epochs` is the total target epoch count, not the number of additional epochs.

```bash
./build/bin/llama-finetune-qlora \
  --model ~/qwen3-1.7b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --resume ~/adapter.gguf.epoch1.ckpt10.gguf \
  --lora-out ~/adapter.gguf \
  -c 4096 -b 4096 -ub 512 \
  --save-every 10 --epochs 3
```

Checkpoint GGUFs contain the LoRA weights and training position. Optimizer moments are not stored, so the optimizer starts fresh after resume. Older checkpoints without embedded position metadata are supported when their original `.epochE.ckptW.gguf` filename is preserved. With `--shuffle-dataset`, completed epoch shuffles are replayed deterministically before continuing at the next window after W.

### VRAM vs. step-time tradeoff

Step time and VRAM both scale linearly with `-ub`:

| Model | `-ub` | VRAM | Step time (approx) |
|---|---|---|---|
| 1.7B Q4_K_M | 512 | ~18 GB | ~120 s (OOM on 16 GB) |
| 1.7B Q4_K_M | 128 | ~6 GB | ~30 s |
| 15B Q4_K_M | 1024 | ~11 GB | ~60 s |

Use `-c` equal to your target sequence length. More context = more windows per sample = more steps per epoch. Reducing `-c` reduces total training time proportionally.

### Default LoRA targets

llama.cpp uses **internal GGUF tensor names**, not HuggingFace names:

| llama.cpp internal | HuggingFace equivalent | Status |
|---|---|---|
| `attn_q` | `q_proj` | ✅ default target, trainable |
| `attn_output` | `o_proj` | ✅ default target, trainable |
| `ffn_gate` | `gate_proj` | ✅ default target, trainable |
| `ffn_up` | `up_proj` | ✅ default target, trainable |
| `ffn_down` | `down_proj` | ✅ default target, trainable |
| `attn_k` | `k_proj` | ❌ not in defaults — zero gradient (KV scatter via SET_ROWS) |
| `attn_v` | `v_proj` | ❌ not in defaults — zero gradient (KV scatter via SET_ROWS) |
| `ssm_in` | `in_proj` | ❌ not in defaults — zero gradient (SSM_SCAN no backward) |
| `ssm_out` | `out_proj` | ❌ not in defaults — zero gradient (SSM_SCAN no backward) |

**MoE models:** Expert tensors (`*_exps`) are excluded regardless of `--lora-targets`. The quantized expert weights are frozen (stop-gradient), but LoRA on the dense FFN layers (`ffn_gate`, `ffn_up`, `ffn_down`) works — backward via `MUL_MAT_ID` + `OUT_PROD_ID`.

### Dataset format (JSONL)

**Chat format** (loss on response only; use `--train-on-prompt` for all tokens):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

**Prompt/response** (loss on response only):
```json
{"prompt": "What is the capital of France?", "response": "Paris."}
```

**Plain text** (loss on all tokens):
```json
{"text": "The quick brown fox."}
```

**With reward** (Phase 2 — scales gradient by reward):
```json
{"prompt": "...", "response": "...", "reward": 0.85}
```

Rewards are normalized per epoch: clipped to `[-1, 1]`, then min-max scaled to `[0, 1]`. Reward 0 = sample ignored; reward 1 = full gradient.

### Verify and use the adapter

```bash
# Hot-load for inference (no merge needed)
./build/bin/llama-cli --model base.gguf --lora adapter.gguf -p "Hello"

# Merge into base model
./build/bin/llama-export-lora \
  --model base.gguf --lora adapter.gguf --output merged.gguf
```

---

## Phase 2 — Reward-Weighted SFT

Built into `llama-finetune-qlora`. When the dataset contains a `reward` or `score` field, the cross-entropy loss for that sample is scaled by the reward before backprop. No extra flags needed — detection is automatic.

---

## Phase 3 — GRPO (Online RL via IPC)

`llama-finetune-qlora --grpo-mode` implements a full GRPO training loop where the Python process owns prompt sampling and reward scoring, and the C++ process owns model state, generation, and gradient updates.

### Quick start

```bash
python3 examples/qlora_training/grpo_example.py \
    --model  ~/qwen3-1.7b-q4_k_m.gguf \
    --lora-out ~/grpo-adapter.gguf \
    --rank 16 --n-steps 200 --n-gen 8
```

For verbose output (includes IPC message trace):

```bash
python3 examples/qlora_training/grpo_example.py \
    --model ~/qwen3-1.7b-q4_k_m.gguf \
    --lora-out ~/grpo-adapter.gguf \
    --verbose
```

Resume from a checkpoint:

```bash
python3 examples/qlora_training/grpo_example.py \
    --model ~/qwen3-1.7b-q4_k_m.gguf \
    --resume   ~/grpo-adapter.ckpt50.gguf \
    --lora-out ~/grpo-adapter.gguf
```

### GRPO-specific flags

| Flag | Default | Description |
|---|---|---|
| `--grpo-mode` | off | Enable GRPO IPC mode |
| `--n-gen` | `8` | Rollouts per prompt |
| `--n-steps` | `500` | Total GRPO steps |
| `--grpo-temp` | `0.8` | Sampling temperature for rollouts |
| `--grpo-max-tokens` | `512` | Max tokens per generation |

All standard flags (`--lora-rank`, `-lr`, `-c`, `-ngl`, `--save-every`, etc.) work in GRPO mode too. `--train-file` is **not** required in GRPO mode.

### IPC protocol

The protocol is line-based over stdout (C++ → Python) and stdin (Python → C++). All non-protocol C++ output (timing, debug, model logs) goes to **stderr** and never contaminates the protocol channel.

**C++ → Python (stdout):**

| Line | When |
|---|---|
| `[QLORA:READY]` | Process initialised, model loaded |
| `[QLORA:PROMPT_REQ:<step>]` | C++ requests the prompt for step N |
| `[QLORA:GEN:<k>/<n>] <text>` | One generation (newlines escaped as `\n`) |
| `[QLORA:REWARD_REQ:<n>]` | C++ requests N reward scores |
| `[QLORA:PROGRESS] step=X/Y loss=Z epoch=A/B` | After each weight update |
| `[QLORA:CHECKPOINT] <path>` | After saving a checkpoint |
| `[QLORA:DONE] final_loss=X` | Training complete |
| `[QLORA:ERROR] <message>` | Fatal error |

**Python → C++ (stdin):**

| Line | Meaning |
|---|---|
| `PROMPT <escaped_text>` | Send prompt for the most recent `PROMPT_REQ` |
| `REWARD <r1> <r2> … <rN>` | Send N advantage scores in `[0, 1]` range |
| `STOP` | Request graceful shutdown after current step |

**Text encoding:** newlines in generation text are escaped as the two-character sequence `\n`; backslashes are doubled. Use `unescape()` from `grpo_example.py` (or any equivalent) to recover the original text.

### Writing your own driver

`grpo_example.py` contains two functions you replace with your own logic:

```python
def get_prompt(step: int) -> str:
    """Return the training prompt for step N."""
    ...

def score_generations(prompt: str, generations: List[str]) -> List[float]:
    """Score each generation. Any numeric range — will be normalised."""
    ...
```

The IPC helpers (`escape`, `unescape`, `parse_ipc`, `read_ipc`, `write_cmd`, `wait_for`, `normalise_rewards`) are standalone and have no external dependencies — copy them into your own project if needed.

### Training loop diagram

```
Python                         C++ (llama-finetune-qlora --grpo-mode)
  │                                │
  │◄──── [QLORA:READY] ────────────┤  model loaded
  │                                │
  │  ┌─────────────────────────────┤
  │  │ for each step:              │
  │  │   ◄── PROMPT_REQ:N ─────────┤
  │  │   ──► PROMPT <text> ────────►  generate n_gen rollouts
  │  │        ◄── GEN:1/n <text> ──┤
  │  │        ◄── GEN:2/n <text> ──┤
  │  │        ...                  │
  │  │        ◄── GEN:n/n <text> ──┤
  │  │   ◄── REWARD_REQ:n ─────────┤
  │  │   (score generations)       │
  │  │   ──► REWARD a1 a2 … an ────►  one backward + AdamW step
  │  │   ◄── PROGRESS step=N/M … ──┤
  │  └─────────────────────────────┤
  │                                │
  │◄──── [QLORA:DONE] ─────────────┤  adapter saved
```

---

## Known Limitations & Optimization Roadmap

### Current limitations

**1. Full backprop through frozen quantized layers**
Every backward step dequantizes all frozen Q4_K_M weight tensors to compute activation gradients (needed to propagate loss from the output back to each LoRA layer). For a 28-layer 1.7B model at `-ub 512`, this is ~280 dequantizing matmuls per step → step time is 3–5× slower than inference.

**2. Activation VRAM** *(partially addressed by `--grad-checkpoint`)*
All forward activations are kept in VRAM throughout the backward pass. VRAM ≈ `model + KV + n_layers × hidden × n_ubatch × 10 × 4B + 2 × lora_params × 4B`. Reducing `-ub` reduces VRAM linearly. Use `--grad-checkpoint 48` to prevent the allocator from reusing intermediate activation buffers during backward, which cuts peak activation VRAM at near-zero compute cost.

**3. Full backprop through all layers** *(partially addressed by `--freeze-layers`)*
Gradients propagate through all layers that have LoRA adapters. Use `--freeze-layers N` to skip LoRA allocation for blk.0..N-1 — those layers receive no gradient (the `grads_needed` pruner already skips their backward ops automatically). Only the top (total_layers - N) layers are trained.

### Optimization roadmap

| Priority | Optimization | Expected gain | Status |
|---|---|---|---|
| ✅ Done | **`--freeze-layers N`** — no LoRA on first N layers; backward auto-pruned | Proportional to N/total | Implemented |
| ✅ Done | **`--grad-checkpoint N`** — keep every Nth activation alive through backward | Reduces peak activation VRAM | Implemented |
| ✅ Done | **`--train-on-prompt`** — compute loss on prompt tokens too | Configurable loss target | Implemented |
| ✅ Done | **`--shuffle-dataset`** — shuffle windows each epoch | Better convergence | Implemented |
| ✅ Done | **BOS separators** — insert BOS between concatenated samples | Correct cross-sample boundaries | Implemented |
| ✅ Done | **Per-epoch loss summary** — log train/val loss after each epoch | Observability | Implemented |
| ✅ Done | **`MUL_MAT_ID` backward** — LoRA on MoE dense FFN layers; `OUT_PROD_ID` for scattered outer product | Unlocks Mixtral/Nemotron-MoE | Implemented |
| Done | **Quantized `OUT_PROD`** - dequantize on GPU + cuBLAS/hipBLAS for backward matmul | Full GPU training (no CPU fallback) | Implemented |
| ✅ Done | **Reuse `ctx_compute_opt`** — allocate tensor metadata context once, `ggml_reset()` across ubatches | Eliminate ~0.5 s/step overhead | Implemented |
| ❌ Skip | **Static training graphs** — KV mask shape changes per ubatch (`n_kv` grows); graph topology not static | Would need KV cache redesign | Not feasible |
| Low | **`SSM_SCAN/CONV` backward** — enable LoRA on Mamba SSM layers | Unlocks NemotronH SSM layers | Planned |
| Low | **GELU backward** — implement `ggml_gelu_back` kernel (UNARY + GLU) | Support GPT-2/Phi-style models | Planned (needs new CUDA/CPU kernels) |

---

## Implementation notes (for developers)

### Modified llama.cpp files

| File | Change |
|---|---|
| `ggml/src/ggml.c` | Backward graph fixes: `GET_ROWS` 3D, `SET_ROWS`, `MUL_MAT_ID`, `SSM_SCAN/CONV`, `FLASH_ATTN_EXT` all stop gradient; inplace-op assert → warn+skip |
| `src/llama-context.cpp` | `opt_init`: scheduler and graph sized with inflated capacity before `ggml_opt_init`; `opt_epoch_iter`: per-ubatch timing instrumentation; reward scaling via `g_reward_weights` TLS |
| `src/llama-adapter.cpp` | Repack-buft fallback for LoRA tensors: tries device-native buft before CPU |
| `common/common.h` | Added `save_every`, `lora_resume`, `lora_freeze_layers`, `grad_checkpoint_interval`, `train_on_prompt`, `shuffle_dataset` fields |
| `common/arg.cpp` | Added `--save-every`, `--resume`, `--freeze-layers`, `--grad-checkpoint`, `--train-on-prompt`, `--shuffle-dataset` arguments |
| `include/llama.h` | Added `llama_opt_set_reward_weights()` and `llama_opt_epoch_range()`; `grad_checkpoint_interval` in `llama_opt_params`; `shuffle` param in `llama_opt_epoch` |
| `ggml/src/ggml-cuda/out-prod.cu` | Shared CUDA/HIP `OUT_PROD` with quantized src0 (dequantize on GPU + cuBLAS/hipBLAS); `OUT_PROD_ID` for MoE backward |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_op` for quantized `OUT_PROD` and `OUT_PROD_ID`; CPU-resident ids fix in `mul_mat_id` |
| `ggml/include/ggml-opt.h` | Added `grad_checkpoint_interval` to `ggml_opt_params` |
| `ggml/src/ggml-opt.cpp` | Gradient checkpointing: marks every Nth forward node `GGML_TENSOR_FLAG_OUTPUT` before backward build |

### Key invariants

- `params.use_mmap = false` — forced; mmap'd tensors can't have data written back
- `params.flash_attn_type = DISABLED` — no backward impl for flash attention
- `params.warmup = false` — warmup runs inference with PARAM tensors → segfault
- `params.cache_type_k = F32` — training requires F32 KV (or BF16 with `--cache-type-k bf16`)
- LoRA A/B tensors are marked `PARAM` via `ggml_set_param` on the tensors loaded by `llama_adapter_lora_init`, not on the pre-init scratch tensors in `lt.buf`
- The adapter GGUF is pre-saved and loaded via `params.lora_adapters` BEFORE `common_init_from_params` so that `sched_reserve` includes LoRA graph nodes in its sizing

### Why opt_init inflation matters

`ggml_opt_init` captures `sched.get()` at construction time. The backward graph (`gb_grad`, `gb_opt`) is ~3–5× larger than the forward graph in node count. If the scheduler hash_set is sized only for the forward graph, `ggml_backend_sched_alloc_graph` on the backward graph will overflow it. We recreate `sched` with `inflated = fwd_nodes × 4` slots BEFORE calling `ggml_opt_init`.

### Reward weighting implementation

`llama_opt_set_reward_weights(weights, n)` sets thread-local `g_reward_weights`. In `opt_epoch`, each window reads `g_reward_weights[idata]` and passes it as `reward_scale` to `opt_epoch_iter`. Inside the iter loop, instead of writing `1.0f` for the correct token's label position in the cross-entropy label tensor, it writes `reward_scale`. Since cross-entropy loss = `-mean(label × log(softmax(logit)))`, scaling the label scales both loss and gradient identically.
