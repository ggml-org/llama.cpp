#!/usr/bin/env python3
"""Overfit-First ML Debugging Protocol for Router Predictor.

Architecture:
  Layer-Decoupled Vectorized MLP:
  - 48 independent per-layer MLPs running in a single batched GEMM
  - Context: 4 previous steps multi-hot concatenated (input_dim = 4 * 512 = 2048)
  - Hidden: 256 with ReLU
  - Output: 512 logits per layer
  - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
  - Loss: BCEWithLogitsLoss(pos_weight=40.0)
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    # If run outside the uv venv, re-exec with uv venv python if available
    venv_py = "/tmp/uv_venv/bin/python"
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise


class LayerDecoupledMLP(nn.Module):
    def __init__(self, n_layers=48, in_dim=2048, hidden_dim=256, out_dim=512):
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


def make_dataset(A, K=4, n_exp=512):
    T, L, _ = A.shape
    B = T - K
    X = torch.zeros(B, L, K * n_exp, dtype=torch.float32)
    Y = torch.zeros(B, L, n_exp, dtype=torch.float32)
    for t in range(K, T):
        b_idx = t - K
        for k in range(K):
            for l in range(L):
                X[b_idx, l, k * n_exp + A[t - K + k, l]] = 1.0
        for l in range(L):
            Y[b_idx, l, A[t, l]] = 1.0
    return X, Y


def eval_recall(logits, A, K=4):
    topk20 = torch.topk(logits, 20, dim=-1).indices
    T, L, _ = A.shape
    res = {}
    for k_eval in [10, 16, 20]:
        top_k = topk20[:, :, :k_eval]
        hit, tot = 0, 0
        for l in range(L):
            target = torch.from_numpy(A[K:T, l].astype(np.int64))
            hits_l = (top_k[:, l, :, None] == target[:, None, :]).any(dim=-1).sum().item()
            hit += hits_l
            tot += target.numel()
        res[k_eval] = 100.0 * hit / tot
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="/tmp/bg.ids.u16")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=40.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(12)

    # Load trace
    raw = np.fromfile(args.trace, dtype=np.uint16)
    all_steps = raw.reshape(-1, 48, 10)
    # Decode steps start after prefill (index 6)
    P1 = all_steps[6 : 6 + 512]
    P2 = all_steps[6 + 512 : 6 + 1024]

    os.makedirs("traces/qwen38", exist_ok=True)
    P1.tofile("traces/qwen38/trace_512.ids.u16")
    # Also save u8 modulo 256 for systems requiring u8
    (P1 % 256).astype(np.uint8).tofile("traces/qwen38/trace_512.ids.u8")

    X1, Y1 = make_dataset(P1, K=args.ctx)
    X2, Y2 = make_dataset(P2, K=args.ctx)

    model = LayerDecoupledMLP(n_layers=48, in_dim=args.ctx * 512, hidden_dim=args.hidden_dim, out_dim=512)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: Layer-Decoupled MLPs (48 layers)")
    print(f"Parameters: {n_params:,} ({n_params * 4 / 1024 / 1024:.2f} MB fp32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

    print(f"\n--- Training on Prompt 1 (512 steps, K={args.ctx}) ---")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits1 = model(X1)
        loss = criterion(logits1, Y1)
        loss.backward()
        optimizer.step()

        if epoch in [10, 20, 50, 100, 150, 200] or epoch == args.epochs:
            with torch.no_grad():
                r1 = eval_recall(logits1, P1, K=args.ctx)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | R@10: {r1[10]:.2f}% | R@16: {r1[16]:.2f}% | R@20: {r1[20]:.2f}%")

    with torch.no_grad():
        r1_final = eval_recall(model(X1), P1, K=args.ctx)
        r2_final = eval_recall(model(X2), P2, K=args.ctx)

    print("\n--- Final Results ---")
    print(f"Prompt 1 (Train): R@10={r1_final[10]:.2f}% | R@16={r1_final[16]:.2f}% | R@20={r1_final[20]:.2f}%")
    print(f"Prompt 2 (Test):  R@10={r2_final[10]:.2f}% | R@16={r2_final[16]:.2f}% | R@20={r2_final[20]:.2f}%")
    print(f"Generalization Gap: R@10 Gap={r1_final[10]-r2_final[10]:.2f}% | R@16 Gap={r1_final[16]-r2_final[16]:.2f}% | R@20 Gap={r1_final[20]-r2_final[20]:.2f}%")


if __name__ == "__main__":
    main()
