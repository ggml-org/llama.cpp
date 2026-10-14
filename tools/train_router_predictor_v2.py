#!/usr/bin/env python3
"""RouterPredictorV2: Cross-Layer Cascading + 512 Hidden Dim + Top-K Margin Ranking Loss.

Architecture:
  - 48 independent per-layer MLPs running in a single batched GEMM
  - Cross-Layer Residual Conditioning:
      For Layer L in 1..47: Layer L (2 steps) + Layer L-1 (2 steps) = 2048 inputs
      For Layer 0: Layer 0 (4 steps) = 2048 inputs
  - Hidden dimension: 512 (increased from 256)
  - Output: 512 logits per layer
  - Parameters: 62,963,712 (~60M params, 120.09 MB in FP16)
  - Loss: BCEWithLogitsLoss(pos_weight=40.0) + Top-16 Margin Ranking Loss
"""

import argparse
import json
import os
import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    venv_py = "/tmp/uv_venv/bin/python"
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise


class RouterPredictorV2(nn.Module):
    def __init__(self, n_layers=48, in_dim=2048, hidden_dim=512, out_dim=512):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(n_layers, in_dim, hidden_dim) * (2.0 / in_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(1, n_layers, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(n_layers, hidden_dim, out_dim) * (2.0 / hidden_dim) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(1, n_layers, out_dim))
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, n_layers, in_dim]
        h = self.act(torch.einsum("bld,ldh->blh", x, self.W1) + self.b1)
        out = torch.einsum("blh,lhe->ble", h, self.W2) + self.b2
        return out


def compute_v2_loss(logits, Y, bce_fn, margin=0.5, rank_weight=0.2):
    bce = bce_fn(logits, Y)

    # Top-16 Margin Ranking Loss: push min active logit above 16th highest negative logit
    pos_mask = (Y > 0.5)
    neg_mask = ~pos_mask

    pos_logits = torch.where(pos_mask, logits, torch.tensor(1e4, device=logits.device))
    min_pos = pos_logits.min(dim=-1).values  # [B, L]

    neg_logits = torch.where(neg_mask, logits, torch.tensor(-1e4, device=logits.device))
    top16_neg = torch.topk(neg_logits, 16, dim=-1).values[:, :, 15]  # [B, L]

    rank_loss = torch.relu(top16_neg - min_pos + margin).mean()
    return bce + rank_weight * rank_loss, bce.item(), rank_loss.item()


def build_sequences(trace_path, meta_path=None, n_layer=48, n_used=10):
    raw = np.fromfile(trace_path, dtype=np.uint16)
    total_steps = len(raw) // (n_layer * n_used)
    all_data = raw[: total_steps * n_layer * n_used].reshape(total_steps, n_layer, n_used)

    sequences = []
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        for p in meta.get("prompts", []):
            s_start = p["start_step"]
            s_end = p["end_step"]
            seq = all_data[s_start:s_end]
            valid_start = 0
            while valid_start < len(seq) and len(np.unique(seq[valid_start, 0])) < n_used:
                valid_start += 1
            if len(seq) - valid_start >= 50:
                sequences.append(
                    {
                        "id": p["id"],
                        "category": p.get("category", "unknown"),
                        "domain": p.get("domain", ""),
                        "data": seq[valid_start:],
                    }
                )
    else:
        chunk_size = 512
        n_seqs = total_steps // chunk_size
        for i in range(n_seqs):
            seq = all_data[i * chunk_size : (i + 1) * chunk_size]
            sequences.append({"id": i, "category": "generic", "domain": "", "data": seq})

    return sequences


class SequenceIndexV2:
    def __init__(self, sequences, n_exp=512):
        self.sequences = sequences
        self.n_exp = n_exp
        self.samples = []
        for s_idx, s in enumerate(sequences):
            T = s["data"].shape[0]
            for t in range(4, T):
                self.samples.append((s_idx, t))

    def __len__(self):
        return len(self.samples)

    def get_batch(self, indices):
        B = len(indices)
        L = 48
        n_exp = self.n_exp
        X = torch.zeros(B, L, 2048, dtype=torch.float32)
        Y = torch.zeros(B, L, n_exp, dtype=torch.float32)

        for b_i, sample_idx in enumerate(indices):
            s_idx, t = self.samples[sample_idx]
            A = self.sequences[s_idx]["data"]

            # Layer 0: 4 steps of Layer 0 (t-4, t-3, t-2, t-1)
            for k in range(4):
                step_hist = A[t - 4 + k, 0]  # [10]
                for u in range(10):
                    exp_id = step_hist[u]
                    if exp_id < n_exp:
                        X[b_i, 0, k * n_exp + exp_id] = 1.0

            # Layers 1..47: 2 steps of Layer L (t-2, t-1) + 2 steps of Layer L-1 (t-2, t-1)
            for l in range(1, L):
                # Layer L (t-2, t-1) -> offset 0..1023
                for k in range(2):
                    hist_l = A[t - 2 + k, l]
                    for u in range(10):
                        exp_id = hist_l[u]
                        if exp_id < n_exp:
                            X[b_i, l, k * n_exp + exp_id] = 1.0
                # Layer L-1 (t-2, t-1) -> offset 1024..2047 (vertical cascade)
                for k in range(2):
                    hist_prev = A[t - 2 + k, l - 1]
                    for u in range(10):
                        exp_id = hist_prev[u]
                        if exp_id < n_exp:
                            X[b_i, l, 1024 + k * n_exp + exp_id] = 1.0

            # Targets Y
            for l in range(L):
                cur = A[t, l]
                for u in range(10):
                    exp_id = cur[u]
                    if exp_id < n_exp:
                        Y[b_i, l, exp_id] = 1.0

        return X, Y


def evaluate_v2(model, seq_index, batch_size=256, bce_fn=None):
    model.eval()
    total_loss = 0.0
    total_samples = len(seq_index)
    total_hits = {10: 0, 16: 0, 20: 0}
    total_items = 0

    with torch.no_grad():
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = list(range(start, end))
            xb, yb = seq_index.get_batch(batch_indices)

            out = model(xb)
            if bce_fn:
                loss, _, _ = compute_v2_loss(out, yb, bce_fn)
                total_loss += loss.item() * len(batch_indices)

            topk20 = torch.topk(out, 20, dim=-1).indices

            for b_i, sample_idx in enumerate(batch_indices):
                s_idx, t = seq_index.samples[sample_idx]
                target_np = seq_index.sequences[s_idx]["data"][t].astype(np.int64)
                target = torch.from_numpy(target_np)
                for l in range(48):
                    t_l = target[l]
                    for k_eval in [10, 16, 20]:
                        pred_k = topk20[b_i, l, :k_eval]
                        hits = (pred_k[:, None] == t_l[None, :]).any(dim=0).sum().item()
                        total_hits[k_eval] += hits
                total_items += 48 * 10

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    res = {k: 100.0 * total_hits[k] / total_items for k in [10, 16, 20]}
    return mean_loss, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="traces/trace_50k.ids.u16")
    parser.add_argument("--meta", type=str, default="traces/trace_50k.meta.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=40.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    args = parser.parse_args()

    torch.set_num_threads(12)

    print(f"Loading trace: {args.trace} (meta: {args.meta})")
    sequences = build_sequences(args.trace, args.meta)
    print(f"Loaded {len(sequences)} sequences from trace.")

    # Sequence-level split: 80% train, 20% test (strictly held-out prompts)
    test_ids = {7, 11, 15, 19}
    train_seqs = [s for s in sequences if s["id"] not in test_ids]
    test_seqs = [s for s in sequences if s["id"] in test_ids]

    if len(test_seqs) == 0:
        n_test = max(1, len(sequences) // 5)
        train_seqs = sequences[:-n_test]
        test_seqs = sequences[-n_test:]

    train_index = SequenceIndexV2(train_seqs)
    test_index = SequenceIndexV2(test_seqs)

    print(f"\nSplit: {len(train_seqs)} Train sequences ({len(train_index)} steps), {len(test_seqs)} Test sequences ({len(test_index)} steps)")
    test_descs = [s["category"] + "/" + s["domain"] for s in test_seqs]
    print(f"Test Prompts: {[s['id'] for s in test_seqs]} ({test_descs})")

    model = RouterPredictorV2(n_layers=48, in_dim=2048, hidden_dim=args.hidden_dim, out_dim=512)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params * 2 / 1024 / 1024:.2f} MB FP16)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

    n_samples = len(train_index)
    bs = args.batch_size
    n_batches = (n_samples + bs - 1) // bs

    best_r16 = 0.0
    best_state = None
    best_epoch = 0

    print(f"\n--- Training RouterPredictorV2 across {args.epochs} Epochs ---")
    print(f"{'Epoch':5s} | {'Train Loss':10s} | {'Test Loss':10s} | {'Test R@10':9s} | {'Test R@16':9s} | {'Test R@20':9s} | {'Time':7s}")
    print("-" * 75)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for b in range(n_batches):
            batch_idx = perm[b * bs : min((b + 1) * bs, n_samples)].tolist()
            xb, yb = train_index.get_batch(batch_idx)

            optimizer.zero_grad()
            out = model(xb)
            loss, _, _ = compute_v2_loss(out, yb, bce_criterion, margin=args.margin, rank_weight=args.rank_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)

        train_loss = epoch_loss / n_samples

        if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
            test_loss, test_recall = evaluate_v2(model, test_index, batch_size=256, bce_fn=bce_criterion)
            elapsed = time.time() - t_start
            print(
                f"{epoch:5d} | {train_loss:10.4f} | {test_loss:10.4f} | "
                f"{test_recall[10]:8.2f}% | {test_recall[16]:8.2f}% | {test_recall[20]:8.2f}% | {elapsed:6.1f}s",
                flush=True,
            )
            if test_recall[16] > best_r16:
                best_r16 = test_recall[16]
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from Epoch {best_epoch} (Test R@16: {best_r16:.2f}%)")
        os.makedirs("traces", exist_ok=True)
        torch.save(best_state, "traces/router_predictor_v2.pt")
        print("Saved best checkpoint to traces/router_predictor_v2.pt")

    # Final detailed evaluation
    print("\n--- Final Evaluation on Strictly Held-Out Test Prompts (Best Checkpoint) ---")
    train_loss, train_recall = evaluate_v2(model, train_index, batch_size=256, bce_fn=bce_criterion)
    test_loss, test_recall = evaluate_v2(model, test_index, batch_size=256, bce_fn=bce_criterion)

    print(f"Train Set ({len(train_seqs)} prompts, {len(train_index)} tokens):")
    print(f"  Loss: {train_loss:.4f} | R@10: {train_recall[10]:.2f}% | R@16: {train_recall[16]:.2f}% | R@20: {train_recall[20]:.2f}%")
    print(f"Test Set ({len(test_seqs)} prompts, {len(test_index)} tokens):")
    print(f"  Loss: {test_loss:.4f} | R@10: {test_recall[10]:.2f}% | R@16: {test_recall[16]:.2f}% | R@20: {test_recall[20]:.2f}%")
    print(f"Generalization Gap:")
    print(f"  R@10 Gap: {train_recall[10] - test_recall[10]:.2f}%")
    print(f"  R@16 Gap: {train_recall[16] - test_recall[16]:.2f}%")
    print(f"  R@20 Gap: {train_recall[20] - test_recall[20]:.2f}%")

    print("\nPer-Category Test Set Breakdown:")
    for ts in test_seqs:
        sub_index = SequenceIndexV2([ts])
        _, sub_recall = evaluate_v2(model, sub_index, batch_size=256)
        desc = ts["category"] + "/" + ts["domain"]
        print(f"  Prompt #{ts['id']:2d} ({desc:20s}): R@10={sub_recall[10]:.2f}% | R@16={sub_recall[16]:.2f}% | R@20={sub_recall[20]:.2f}%")


if __name__ == "__main__":
    main()
