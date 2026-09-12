# MoE Expert Pruning Tools for NemotronH

REAP-style expert pruning for `NVIDIA-Nemotron-3-Nano-30B-A3B` (and other
NemotronH MoE models), implemented in two complementary ways:

1. **`tools/expert-profile/`** — C++ profiler built into llama.cpp, collects
   REAP scores directly from GGUF inference via the ggml eval callback.
2. **`tools/moe-pruning/`** (this directory) — Python scripts to prune the model
   using the collected scores, either on a GGUF file directly or on a
   HuggingFace BF16 checkpoint.

---

## Inspiration & Prior Art

This work is a direct implementation of the **REAP** saliency criterion
introduced in:

> **REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression**
> Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, Vithursan Thangarasa
> Cerebras Research, 2025
> arXiv: https://arxiv.org/abs/2510.13999
> Code:  https://github.com/CerebrasResearch/reap

The REAP score for expert `j` is (Equation 9 of the paper):

```
REAP(j) = mean_{t : j ∈ topk(t)} [ g_j(t) · ‖f_j(t)‖₂ ]
```

where `g_j(t)` is the router gate weight and `f_j(t)` is the expert FFN output
(pre-weighting) for token `t`. Experts with the lowest REAP score contribute
least to the layer output and are pruned first.

The original REAP repo targets HuggingFace models via PyTorch hooks on
standard architectures (Qwen3-MoE, Mixtral, DeepSeek-V2, Llama-4, …).

**What we added / adapted:**

- `tools/expert-profile/expert-profile.cpp` — llama.cpp C++ implementation
  of REAP that intercepts `ffn_moe_topk`, `ffn_moe_weights`, and `ffn_moe_down`
  tensors via `ggml_backend_eval_callback`, enabling REAP profiling on any
  GGUF-quantised model (Q4_K_M, Q6_K, etc.) without needing full BF16 VRAM.

- `gguf_prune.py` — prunes the GGUF file **directly**, slicing the expert axis
  of the stacked weight tensors (`ffn_up_exps`, `ffn_down_exps`, `ffn_gate_inp`,
  `ffn_exp_probs_b`) and patching `{arch}.expert_count` in the metadata.
  Quantised blocks are preserved as raw bytes — no dequantise/requantise step.

- `nemotron_reap.py` — HuggingFace-based alternative: profiles with 4-bit NF4
  on GPU (phase 1) and prunes the BF16 checkpoint on CPU (phase 2). Adds
  NemotronH (`NemotronHForCausalLM`) support that the original REAP repo does
  not have.

---

## Recommended Workflow (low-VRAM, e.g. RTX 4060 Ti 16 GB)

```
┌─────────────────────────────────────────────┐
│  Phase 1 — Profile  (GPU, GGUF Q4, ~15 GB)  │
│                                             │
│  llama-expert-profile                       │
│    -m nemotron-Q4_K_M.gguf                  │
│    --jsonl sample_calibration.jsonl         │
│    --output expert_stats.json               │
│    -ngl 99 --ctx-size 2048                  │
└───────────────────┬─────────────────────────┘
                    │ expert_stats.json
┌───────────────────▼─────────────────────────┐
│  Phase 2 — Prune  (CPU, pure Python, ~2 GB) │
│                                             │
│  python gguf_prune.py                       │
│    --input  nemotron-Q4_K_M.gguf            │
│    --stats  expert_stats.json               │
│    --output nemotron-pruned-26e.gguf        │
│    --keep_ratio 0.20   # 26/128 experts     │
└─────────────────────────────────────────────┘
```

At 20 % keep ratio a ~22 GB Q4_K_M becomes ~4.5 GB.

---

## Files

| File | Description |
|---|---|
| `gguf_prune.py` | GGUF-native pruner — no GPU needed, preserves quantisation |
| `nemotron_reap.py` | HF-based pruner — 4-bit GPU profile + CPU BF16 prune |
| `build_expert_profile.sh` | Build script for `llama-expert-profile` |
| `run_nemotron_profile.sh` | Example profiling run |
| `run_prune.sh` | Example pruning run |
| `run_convert_quantize.sh` | Convert HF → GGUF and quantise |
| `analyze_stats.py` | Visualise and compare expert stats JSON files |
| `sample_calibration.jsonl` | Sample calibration data (prompt+response pairs) |
| `expert_stats_reap.json` | Example stats output from expert-profile |
