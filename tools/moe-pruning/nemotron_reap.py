"""
NemotronH Expert Activation Profiler + Pruner
Two-phase: profile with 4-bit on GPU, prune bf16 on CPU.

Usage:
  # Phase 1 - profile
  python nemotron_reap.py profile \
    --model unsloth/Nemotron-3-Nano-30B-A3B \
    --prompts training-data.jsonl \
    --output expert_stats.json

  # Phase 2 - prune
  python nemotron_reap.py prune \
    --model unsloth/Nemotron-3-Nano-30B-A3B \
    --stats expert_stats.json \
    --keep_ratio 0.20 \
    --output ./nemotron-pruned-25e
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # prevent inductor hang during save_pretrained

import json
import argparse
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    import patch_bnb  # noqa: F401 — patches Params4bit.__new__ for transformers 5.x compat
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# ── Tracker ───────────────────────────────────────────────────────────────────

class ExpertActivationTracker:
    def __init__(self, n_experts: int = 128):
        self.n_experts = n_experts
        self.activation_counts  = defaultdict(lambda: np.zeros(n_experts, dtype=np.int64))
        self.activation_weights = defaultdict(lambda: np.zeros(n_experts, dtype=np.float64))
        self.total_tokens = defaultdict(int)
        self._hooks = []

    def register_hooks(self, model):
        count = 0
        for layer_idx, block in enumerate(model.backbone.layers):
            if block.block_type == "moe":
                h = block.mixer.gate.register_forward_hook(self._make_hook(layer_idx))
                self._hooks.append(h)
                count += 1
        print(f"  Hooks attached to {count} MoE layers")

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            topk_indices, topk_weights = output
            idx = topk_indices.detach().cpu().numpy()           # [T, 6]
            wgt = topk_weights.detach().float().cpu().numpy()   # [T, 6]
            T = idx.shape[0]
            self.total_tokens[layer_idx] += T
            np.add.at(self.activation_counts[layer_idx],  idx.flatten(), 1)
            np.add.at(self.activation_weights[layer_idx], idx.flatten(), wgt.flatten())
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_stats(self) -> dict:
        stats = {}
        for layer_idx in sorted(self.activation_counts):
            counts  = self.activation_counts[layer_idx]
            weights = self.activation_weights[layer_idx]
            total   = self.total_tokens[layer_idx]
            freq    = counts / (total + 1e-9)
            avg_w   = np.where(counts > 0, weights / counts, 0.0)
            importance = freq * avg_w
            stats[layer_idx] = {
                "total_tokens":         int(total),
                "activation_counts":    counts.tolist(),
                "activation_frequency": freq.tolist(),
                "avg_weight":           avg_w.tolist(),
                "importance_score":     importance.tolist(),
                "never_activated":      int((counts == 0).sum()),
            }
        return stats

    def print_summary(self, stats, keep_ratio):
        keep_n = max(1, int(self.n_experts * keep_ratio))
        print(f"\n{'='*70}")
        print(f"  PROFILING SUMMARY  |  keep_ratio={keep_ratio:.0%}  |  keeping {keep_n}/128 experts/layer")
        print(f"{'='*70}")
        for li, s in stats.items():
            imp = np.array(s['importance_score'])
            threshold = np.sort(imp)[self.n_experts - keep_n]
            print(
                f"  Layer {li:3d}: "
                f"never_activated={s['never_activated']:3d}/128  "
                f"top_freq={max(s['activation_frequency']):.3f}  "
                f"threshold={threshold:.4f}"
            )
        total_moe = len(stats)
        print(f"\n  MoE layers : {total_moe}")
        print(f"  Kept       : {total_moe * keep_n} experts total")
        print(f"  Pruned     : {total_moe * (self.n_experts - keep_n)} experts total")
        print(f"{'='*70}\n")


# ── Phase 1: Profile ──────────────────────────────────────────────────────────

def cmd_profile(args):
    # Mamba2 layers use Triton kernels — CUDA required.
    # 4-bit NF4 fits in 16GB VRAM (~15GB). We must keep ALL layers on GPU
    # (no CPU spillover) otherwise PCIe transfers make inference unusably slow.
    print(f"\n[Phase 1] Profiling — 4-bit NF4, GPU only")
    print(f"  Model  : {args.model}")
    print(f"  Prompts: {args.prompts}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("  Loading model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},  # force ALL layers onto GPU 0, no CPU spillover
    )
    model.eval()
    print("  Model loaded on GPU.")

    # Load prompt+response pairs
    pairs = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt", "") + "\n" + obj.get("response", "")
            pairs.append(text)
    print(f"  Loaded {len(pairs)} prompt+response pairs")

    tracker = ExpertActivationTracker(n_experts=128)
    tracker.register_hooks(model)

    with torch.no_grad():
        for i, text in enumerate(pairs):
            if i % 100 == 0:
                print(f"  [{i+1}/{len(pairs)}] processing...")
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to("cuda")
            try:
                model(**inputs)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{i+1}] OOM — skipping")
                torch.cuda.empty_cache()

    tracker.remove_hooks()
    stats = tracker.get_stats()
    tracker.print_summary(stats, keep_ratio=args.keep_ratio)

    stats_out = {str(k): v for k, v in stats.items()}
    with open(args.output, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Stats saved → {args.output}")
    print(f"\n[Phase 1] Done. Run 'prune' next (CPU only).")


# ── Phase 2: Prune ────────────────────────────────────────────────────────────

def cmd_prune(args):
    print(f"\n[Phase 2] Pruning — bf16 on CPU")
    print(f"  Model      : {args.model}")
    print(f"  Stats      : {args.stats}")
    print(f"  Keep ratio : {args.keep_ratio:.0%}")
    print(f"  Output     : {args.output}")

    with open(args.stats) as f:
        stats = {int(k): v for k, v in json.load(f).items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("  Loading model in bf16 on CPU — this takes a few minutes...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    keep_n = max(1, int(128 * args.keep_ratio))
    print(f"\n  Pruning to top-{keep_n} experts per MoE layer...\n")

    for layer_idx, block in enumerate(model.backbone.layers):
        if block.block_type != "moe":
            continue

        if layer_idx not in stats:
            print(f"  Layer {layer_idx:3d}: no profiling data — skipping")
            continue

        # Use REAP score if available (from llama.cpp profiler), else fall back to legacy importance_score
        layer_stats = stats[layer_idx]
        if "reap" in layer_stats:
            importance = np.array(layer_stats["reap"])
        else:
            importance = np.array(layer_stats["importance_score"])
        keep_sorted = sorted(np.argsort(importance)[-keep_n:].tolist())
        prune_count = 128 - len(keep_sorted)

        # Prune expert list
        block.mixer.experts = torch.nn.ModuleList(
            [block.mixer.experts[i] for i in keep_sorted]
        )

        # Prune router weights to match new expert indices
        keep_t = torch.tensor(keep_sorted, dtype=torch.long)
        block.mixer.gate.weight = torch.nn.Parameter(
            block.mixer.gate.weight.data[keep_t].clone()
        )
        old_bias = block.mixer.gate.e_score_correction_bias.data[keep_t].clone()
        block.mixer.gate.register_buffer("e_score_correction_bias", old_bias)
        block.mixer.gate.n_routed_experts = keep_n

        never = stats[layer_idx]["never_activated"]
        print(f"  Layer {layer_idx:3d}: kept {keep_n}, pruned {prune_count}  (was {never} never-activated)")

    # Patch top-level config
    model.config.n_routed_experts = keep_n

    # Fix transformers 5.x incompatibility: _tied_weights_keys must be a list of dicts,
    # but the custom NemotronH modeling code sets it as a plain list of strings.
    # _get_tied_weight_keys() calls .keys() on each element → AttributeError.
    # Clear it — lm_head weight tying is not needed for inference on the pruned model.
    for mod in model.modules():
        if isinstance(getattr(mod, '_tied_weights_keys', None), list):
            mod._tied_weights_keys = None

    # Disable torch.compile / inductor before saving — transformers 5.x can trigger
    # torch._inductor.compile_worker during save_pretrained, causing an indefinite hang.
    import os
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch._dynamo.reset()

    print(f"\n  Saving pruned model → {args.output}")
    with torch.no_grad():
        model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"\n[Phase 2] Done.")
    print(f"  Experts per MoE layer : {keep_n}/128")
    print(f"  Next: fine-tune with Unsloth from {args.output}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NemotronH Expert Pruner (REAP-style)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("profile", help="Phase 1: profile expert activations (GPU, 4-bit)")
    p1.add_argument("--model",      default="unsloth/Nemotron-3-Nano-30B-A3B")
    p1.add_argument("--prompts",    required=True)
    p1.add_argument("--output",     default="expert_stats.json")
    p1.add_argument("--keep_ratio", type=float, default=0.20,
                    help="Preview ratio for summary only — does not affect saved stats")
    p1.add_argument("--max_length", type=int,   default=2048)

    p2 = sub.add_parser("prune", help="Phase 2: prune model using saved stats (CPU, bf16)")
    p2.add_argument("--model",      default="unsloth/Nemotron-3-Nano-30B-A3B")
    p2.add_argument("--stats",      default="expert_stats.json")
    p2.add_argument("--keep_ratio", type=float, default=0.20)
    p2.add_argument("--output",     default="./nemotron-pruned")

    args = parser.parse_args()
    if args.cmd == "profile":
        cmd_profile(args)
    elif args.cmd == "prune":
        cmd_prune(args)


if __name__ == "__main__":
    main()
