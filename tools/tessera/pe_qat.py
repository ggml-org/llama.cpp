#!/usr/bin/env python3
"""PE-QAT: parameter-efficient quantization-aware training.

Implements the PE-QAT recipe (ACL 2026 SRW) in pure NumPy.  The base model
weights are frozen; LoRA adapters, SmoothQuant-style per-channel smoothing
factors, and clipping thresholds are trained via STE backprop through
fake-quantization.  ~1.26% of total parameters are trainable at the model
level (the same ratio holds per-layer in the demo).

Recipe (per linear layer):

1. Merge the LoRA delta into the base weight: ``W' = W + (alpha/r) * B @ A``
2. SmoothQuant split: ``x' = x / s`` and ``W_s = W' * s`` so that
   ``W_s @ x' = W' @ x`` exactly (the per-channel factor ``s`` is shared
   along the contraction axis).
3. Clip each output channel of ``W_s`` to ``+/- c * max|W_s[o, :]|`` to
   tame outliers before the per-channel symmetric W4 quantizer.
4. Apply per-channel symmetric W4 fake-quant to ``W_s`` and per-tensor
   symmetric A4 fake-quant to ``x'``.
5. Forward ``y = W_q @ x_q`` and minimise ``0.5 * MSE(y, y_ref)`` plus a
   small prior that keeps ``s`` near its initial value.

Backprop is a single explicit reverse sweep over the forward tape.  The
straight-through estimator (STE) is used for the ``round`` step inside
fake-quantization: the gradient is treated as identity through the
rounding.  We deliberately do not use a general autograd framework -- the
forward is short enough to make the manual backward both readable and
debuggable.

The runtime is intentionally simple.  We operate on a single 2-D linear
layer at a time, just like ``make-awq-layer-bundles.py`` already does for
the AWQ-evolve pass.  A future integration with the DSpark drafter will
wrap this loop around the drafter's transformer block.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# Public schema name for the on-disk policy.  Kept aligned with the naming
# convention used by ``llama.speculative.calibration-policy.v1`` and
# ``llama.tessera.unsloth-bridge.v1`` already in this directory.
PE_QAT_POLICY_SCHEMA = "llama.tessera.pe-qat-policy.v1"

# Symmetric int4 / int4 grid.  ``qmax = 7`` is the common convention; we
# offset the grid by 0.5 so the effective bin centres land on integers.
W4A4_QMIN = -8
W4A4_QMAX = 7


# ---------------------------------------------------------------------------
# Trainable parameter containers
# ---------------------------------------------------------------------------


@dataclass
class LoraLinear:
    """LoRA delta ``(alpha / r) * B @ A`` sitting on top of a frozen weight.

    ``W`` is shape ``(out, in)`` and frozen.  ``A`` is ``(r, in)`` and ``B``
    is ``(out, r)``; both are trainable.  ``B`` is initialised to zero so
    that the initial delta is the zero matrix (a common LoRA convention).
    """

    W: np.ndarray
    A: np.ndarray
    B: np.ndarray
    alpha: float
    rank: int
    name: str = "linear"
    # Gradients; owned by the optimiser step.
    A_grad: np.ndarray = field(default=None)  # type: ignore[assignment]
    B_grad: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.A_grad is None:
            self.A_grad = np.zeros_like(self.A)
        if self.B_grad is None:
            self.B_grad = np.zeros_like(self.B)

    @classmethod
    def init(
        cls,
        W: np.ndarray,
        rank: int = 16,
        alpha: float = 32.0,
        rng: np.random.Generator | None = None,
        name: str = "linear",
    ) -> "LoraLinear":
        rng = rng or np.random.default_rng(0)
        out_features, in_features = W.shape
        scale = math.sqrt(1.0 / in_features)
        A = rng.normal(loc=0.0, scale=scale, size=(rank, in_features)).astype(np.float32)
        B = np.zeros((out_features, rank), dtype=np.float32)
        return cls(W=W.astype(np.float32), A=A, B=B, alpha=alpha, rank=rank, name=name)

    def trainable_params(self) -> list[tuple[str, np.ndarray]]:
        return [("A", self.A), ("B", self.B)]

    def param_count(self) -> int:
        return int(self.A.size + self.B.size)

    def zero_grad(self) -> None:
        self.A_grad.fill(0.0)
        self.B_grad.fill(0.0)


@dataclass
class SmoothQuantScale:
    """Per-input-channel smoothing factor ``s`` (shared along ``in_features``).

    Initialised to a value that equalises the per-channel dynamic range
    between the activations and the merged weight.  The recipe mirrors the
    original SmoothQuant paper:
    ``s_j = max|x[:, j]|^alpha / max|W[:, j]|^(1 - alpha)``.
    """

    s: np.ndarray
    s_init: np.ndarray
    alpha: float = 0.5
    name: str = "smooth"
    s_grad: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.s_grad is None:
            self.s_grad = np.zeros_like(self.s)

    @classmethod
    def init(
        cls,
        W: np.ndarray,
        x: np.ndarray,
        alpha: float = 0.5,
        name: str = "smooth",
    ) -> "SmoothQuantScale":
        w_max = np.max(np.abs(W), axis=0)
        x_max = np.max(np.abs(x), axis=0)
        eps = 1e-8
        s = np.power(np.maximum(x_max, eps), alpha) / np.power(
            np.maximum(w_max, eps), 1.0 - alpha
        )
        s = np.clip(s, 1e-4, 1e4).astype(np.float32)
        return cls(s=s, s_init=s.copy(), alpha=alpha, name=name)

    def trainable_params(self) -> list[tuple[str, np.ndarray]]:
        return [("s", self.s)]

    def param_count(self) -> int:
        return int(self.s.size)

    def zero_grad(self) -> None:
        self.s_grad.fill(0.0)


@dataclass
class ClipThreshold:
    """Per-output-channel clipping ratio ``c`` in ``(0, 1]``.

    The forward operation clips each output channel of the (post-SmoothQuant)
    weight to ``+/- c_o * max|W_s[o, :]|``.  ``c = 1`` means no clipping.
    """

    c: np.ndarray
    name: str = "clip"
    c_grad: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.c_grad is None:
            self.c_grad = np.zeros_like(self.c)

    @classmethod
    def init(cls, n_channels: int, name: str = "clip") -> "ClipThreshold":
        return cls(c=np.ones(n_channels, dtype=np.float32), name=name)

    def trainable_params(self) -> list[tuple[str, np.ndarray]]:
        return [("c", self.c)]

    def param_count(self) -> int:
        return int(self.c.size)

    def zero_grad(self) -> None:
        self.c_grad.fill(0.0)


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------


@dataclass
class AdamW:
    """AdamW optimiser state container.  State is held in parallel NumPy
    arrays so a forward pass can update all trainable parameters in a
    single batch.
    """

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    step: int = 0

    def step_param(self, param: np.ndarray, grad: np.ndarray) -> None:
        if not grad.any():
            return
        self.step += 1
        b1_corr = 1.0 - self.beta1 ** self.step
        b2_corr = 1.0 - self.beta2 ** self.step
        key = id(param)
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        m = self.m[key]
        v = self.v[key]
        m *= self.beta1
        m += (1.0 - self.beta1) * grad
        v *= self.beta2
        v += (1.0 - self.beta2) * (grad * grad)
        m_hat = m / b1_corr
        v_hat = v / b2_corr
        update = m_hat / (np.sqrt(v_hat) + self.eps)
        if self.weight_decay:
            update = update + self.weight_decay * param
        param -= self.lr * update


# ---------------------------------------------------------------------------
# Forward / backward primitives
# ---------------------------------------------------------------------------


def fake_quantize_per_channel(
    x: np.ndarray,
    scale: np.ndarray,
    qmin: int = W4A4_QMIN,
    qmax: int = W4A4_QMAX,
) -> np.ndarray:
    """Per-output-channel symmetric fake-quant with STE.  ``scale`` must be
    shape ``(C,)`` or ``(C, 1)`` so it broadcasts over the rows of ``x``.
    The returned gradient under STE is identity (caller responsibility).
    """
    inv = 1.0 / np.maximum(scale, 1e-12)
    x_scaled = x * inv
    x_int = np.clip(np.round(x_scaled), qmin, qmax)
    return (x_int * scale).astype(np.float32)


def fake_quantize_per_tensor(
    x: np.ndarray,
    scale: float,
    qmin: int = W4A4_QMIN,
    qmax: int = W4A4_QMAX,
) -> np.ndarray:
    """Per-tensor symmetric fake-quant with STE."""
    inv = 1.0 / max(scale, 1e-12)
    x_scaled = x * inv
    x_int = np.clip(np.round(x_scaled), qmin, qmax)
    return (x_int * scale).astype(np.float32)


def clip_per_output_channel(
    W: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip each output row of ``W`` to ``+/- c_o * max|W[o, :]|``.

    Returns ``(W_clip, threshold, mask)`` where ``mask`` is 1 where the
    forward is in the unclipped region and 0 where it was clipped.  The
    backward uses ``mask`` for the straight-through gradient.
    """
    c_used = np.clip(c, 1e-3, 1.0)
    row_max = np.max(np.abs(W), axis=1)
    threshold = (c_used * row_max)[:, None]
    W_clip = np.clip(W, -threshold, threshold)
    mask = (np.abs(W) < threshold).astype(np.float32)
    return W_clip.astype(np.float32), threshold, mask


# ---------------------------------------------------------------------------
# End-to-end training step
# ---------------------------------------------------------------------------


@dataclass
class TrainStats:
    iteration: int
    loss: float
    mse: float
    smooth_prior: float
    trainable_params: int
    total_params: int
    trainable_ratio: float


def pe_qat_forward(
    W: np.ndarray,
    x: np.ndarray,
    lora: LoraLinear,
    smooth: SmoothQuantScale,
    clip: ClipThreshold,
    smooth_prior_weight: float = 1e-3,
) -> dict:
    """Run the PE-QAT forward pass and return a bundle of intermediates.

    The bundle is consumed by ``pe_qat_backward`` to compute gradients.
    """
    lora.zero_grad()
    smooth.zero_grad()
    clip.zero_grad()

    # Frozen BF16 reference output (no quant, no LoRA, no SmoothQuant).
    y_ref = (lora.W @ x.T).astype(np.float32)  # (out, batch)

    # 1. LoRA merge under STE: W_lora = W + scaling * B @ A.
    scaling = lora.alpha / float(lora.rank)
    W_lora = lora.W + scaling * (lora.B @ lora.A)

    # 2. SmoothQuant split: W_s = W_lora * s, x_s = x / s.
    s = smooth.s
    W_s = W_lora * s[None, :]
    x_s = (x / s[None, :]).T  # (in, batch)

    # 3. Per-output-channel clip.
    W_clip, threshold, mask = clip_per_output_channel(W_s, clip.c)

    # 4a. Per-channel symmetric W4 fake-quant.
    w_scale = np.max(np.abs(W_clip), axis=1, keepdims=True) / float(W4A4_QMAX)
    w_scale = np.maximum(w_scale, 1e-12)
    W_q = fake_quantize_per_channel(W_clip, w_scale)

    # 4b. Per-tensor symmetric A4 fake-quant.
    a_scale = float(np.max(np.abs(x_s))) / float(W4A4_QMAX) if x_s.size else 0.0
    a_scale = max(a_scale, 1e-12)
    x_q = fake_quantize_per_tensor(x_s, a_scale)

    # 5. Quantized matmul.
    y_q = W_q @ x_q

    # Loss.
    diff = y_q - y_ref
    mse = float(np.mean(diff * diff))
    log_s = np.log(np.maximum(s, 1e-12))
    log_s_init = np.log(np.maximum(smooth.s_init, 1e-12))
    smooth_prior = float(smooth_prior_weight * np.mean((log_s - log_s_init) ** 2))
    loss = 0.5 * mse + smooth_prior

    return {
        "y_ref": y_ref,
        "y_q": y_q,
        "W_q": W_q,
        "x_q": x_q,
        "W_clip": W_clip,
        "W_s": W_s,
        "W_lora": W_lora,
        "x_s": x_s,
        "x": x,
        "s": s,
        "clip_threshold": threshold,
        "clip_mask": mask,
        "scaling": scaling,
        "a_scale": a_scale,
        "loss": loss,
        "mse": mse,
        "smooth_prior": smooth_prior,
    }


def pe_qat_backward(bundle: dict, lora: LoraLinear, smooth: SmoothQuantScale, clip: ClipThreshold) -> None:
    """Reverse-mode backward pass over the bundle produced by ``pe_qat_forward``."""
    y_q = bundle["y_q"]
    y_ref = bundle["y_ref"]
    W_q = bundle["W_q"]
    x_q = bundle["x_q"]
    W_clip = bundle["W_clip"]
    W_s = bundle["W_s"]
    W_lora = bundle["W_lora"]
    x_s = bundle["x_s"]
    x = bundle["x"]
    s = bundle["s"]
    mask = bundle["clip_mask"]
    scaling = bundle["scaling"]

    # 5 -> 4: d(loss)/d(y_q) = (y_q - y_ref) / N.
    diff = y_q - y_ref
    d_y_q = diff / float(diff.size)

    # Matmul backward.
    d_W_q = d_y_q @ x_q.T  # (out, in)
    d_x_q = W_q.T @ d_y_q  # (in, batch)

    # 4: STE through W4 quant: d(W_clip) = d(W_q).
    d_W_clip = d_W_q
    # 4: STE through A4 quant: d(x_s) = d(x_q).  (x is calibration data, not
    # a parameter; we discard the path back to x itself and only retain the
    # contribution to s.)

    # 3: Clip backward.  d(W_s) = d(W_clip) * mask; d(c) computed below.
    d_W_s = d_W_clip * mask
    # d/d(c_o) clip(W_s, threshold_o) = -d(W_clip) for entries where W_s
    # was clipped.  When clipped, the input is the saturated value
    # sign(W_s) * threshold_o, so d(threshold_o)/d(c_o) = row_max_o and
    # d(L)/d(threshold_o) = -sign(W_s) * d(W_clip).  Therefore
    # d(L)/d(c_o) = -sum_j sign(W_s[o, j]) * d(W_clip)[o, j] * row_max[o]
    # restricted to clipped entries (mask == 0).
    row_max = np.max(np.abs(W_s), axis=1)
    inactive = (1.0 - mask)  # 1 where the input was clipped
    sign_dW = np.sign(W_s) * d_W_clip * inactive
    clip.c_grad += np.sum(sign_dW * row_max[:, None], axis=1)

    # 2: SmoothQuant split backward.
    #   W_s = W_lora * s  =>  d(W_lora) = d(W_s) * s,  d(s) += sum_o d(W_s) * W_lora
    d_W_lora = d_W_s * s[None, :]
    smooth.s_grad += np.sum(d_W_s * W_lora, axis=0)
    #   x_s = x / s  =>  d(x) = d(x_s) / s,  d(s) += sum_b d(x_s) * (-x / s^2)
    # ``d_x_q`` is (in, batch); ``s`` is (in,).  Reshape ``s`` to (in, 1) so
    # it broadcasts along the batch axis.
    s_col = s[:, None]
    d_x = d_x_q / s_col  # (in, batch) -- discarded (frozen calibration data)
    smooth.s_grad += np.sum(d_x_q * (-x.T / (s_col * s_col)), axis=1)

    # 1: LoRA split.  W_lora = W + scaling * B @ A; W is frozen.
    #   d(B) = (1 / scaling) * d(W_lora) @ A.T
    #   d(A) = (1 / scaling) * B.T @ d(W_lora)
    inv_scaling = 1.0 / scaling
    lora.B_grad += inv_scaling * (d_W_lora @ lora.A.T)
    lora.A_grad += inv_scaling * (lora.B.T @ d_W_lora)


def pe_qat_train_step(
    W: np.ndarray,
    x: np.ndarray,
    lora: LoraLinear,
    smooth: SmoothQuantScale,
    clip: ClipThreshold,
    smooth_prior_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TrainStats]:
    """Run a single PE-QAT training step and return reference + quantized outputs."""
    bundle = pe_qat_forward(W, x, lora, smooth, clip, smooth_prior_weight=smooth_prior_weight)
    pe_qat_backward(bundle, lora, smooth, clip)
    trainable = lora.A.size + lora.B.size + smooth.s.size + clip.c.size
    total = lora.W.size + trainable
    stats = TrainStats(
        iteration=-1,
        loss=bundle["loss"],
        mse=bundle["mse"],
        smooth_prior=bundle["smooth_prior"],
        trainable_params=trainable,
        total_params=total,
        trainable_ratio=trainable / total,
    )
    return bundle["y_ref"], bundle["y_q"], bundle["W_q"], stats


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def pe_qat_train(
    W: np.ndarray,
    x: np.ndarray,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    iters: int = 100,
    lr: float = 1e-3,
    smooth_prior_weight: float = 1e-3,
    log_every: int = 10,
    seed: int = 0,
) -> dict:
    """Train a single PE-QAT layer and return the learned policy + loss curve.

    ``W`` is the frozen base weight (out, in).  ``x`` is the calibration
    activations (batch, in).  The returned dict contains the LoRA delta,
    the trained smoothing factors, the trained clip thresholds, and the
    per-iteration loss curve.
    """
    rng = np.random.default_rng(seed)
    lora = LoraLinear.init(W, rank=rank, alpha=alpha, rng=rng)
    smooth = SmoothQuantScale.init(W, x, alpha=0.5)
    clip = ClipThreshold.init(W.shape[0])
    opt = AdamW(lr=lr)

    losses: list[float] = []
    mses: list[float] = []

    for it in range(iters):
        y_ref, y_q, _, stats = pe_qat_train_step(
            W, x, lora, smooth, clip, smooth_prior_weight=smooth_prior_weight,
        )
        opt.step_param(lora.A, lora.A_grad)
        opt.step_param(lora.B, lora.B_grad)
        opt.step_param(smooth.s, smooth.s_grad)
        opt.step_param(clip.c, clip.c_grad)
        # Box-projection on c: the forward clamps c to (0, 1] so values
        # outside that range have no gradient, but AdamW's momentum can
        # still drift the parameter.  Re-project after each step.
        np.clip(clip.c, 1e-3, 1.0, out=clip.c)
        stats.iteration = it
        losses.append(stats.loss)
        mses.append(stats.mse)
        if (it % log_every) == 0 or it == iters - 1:
            print(
                f"  iter {it:4d}  loss={stats.loss:.4e}  mse={stats.mse:.4e}  "
                f"smooth={stats.smooth_prior:.4e}  trainable={stats.trainable_params} "
                f"({100 * stats.trainable_ratio:.2f}%)"
            )

    return {
        "lora_A": lora.A,
        "lora_B": lora.B,
        "s": smooth.s,
        "s_init": smooth.s_init,
        "c": clip.c,
        "W_ref": lora.W,
        "losses": losses,
        "mses": mses,
        "rank": rank,
        "alpha": alpha,
        "iters": iters,
        "lr": lr,
        "smooth_prior_weight": smooth_prior_weight,
        "smooth_alpha": smooth.alpha,
    }


# ---------------------------------------------------------------------------
# Output: policy JSON
# ---------------------------------------------------------------------------


def save_pe_qat_policy(path: Path, result: dict, *, family: str = "attention") -> None:
    """Write a ``llama.tessera.pe-qat-policy.v1`` JSON for the trained layer.

    The schema extends the conventional ``llama.speculative.calibration-policy.v1``
    layout with the four new fields a quantizer needs to apply the PE-QAT
    recipe: ``lora_A``, ``lora_B``, ``per_channel_smooth_s``, and
    ``per_channel_clip_c``.  All arrays are serialised as ``list[float]``
    so the file is diff-friendly.
    """
    policy = {
        "schema": PE_QAT_POLICY_SCHEMA,
        "family": family,
        "rank": result["rank"],
        "alpha": result["alpha"],
        "iters": result["iters"],
        "lr": result["lr"],
        "smooth_prior_weight": result["smooth_prior_weight"],
        "smooth_alpha": result["smooth_alpha"],
        "lora_A": result["lora_A"].tolist(),
        "lora_B": result["lora_B"].tolist(),
        "per_channel_smooth_s": result["s"].tolist(),
        "per_channel_smooth_s_init": result["s_init"].tolist(),
        "per_channel_clip_c": result["c"].tolist(),
        "trainable_params": int(result["lora_A"].size + result["lora_B"].size +
                                result["s"].size + result["c"].size),
        "base_weight_params": int(result["W_ref"].size),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def apply_pe_qat(
    W: np.ndarray,
    x: np.ndarray,
    result: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a trained PE-QAT policy to a (W, x) pair.

    Returns the reference BF16 output and the quantized output.  Used by
    the demo and the verification test to report the BF16-vs-quantized
    MSE without running the training loop.
    """
    A = result["lora_A"]
    B = result["lora_B"]
    s = result["s"]
    c = result["c"]
    scaling = result["alpha"] / float(result["rank"])
    W_lora = W + scaling * (B @ A)
    y_ref = (W_lora @ x.T).astype(np.float32)
    W_s = W_lora * s[None, :]
    x_s = (x / s[None, :]).T
    W_clip, _, _ = clip_per_output_channel(W_s, c)
    w_scale = np.max(np.abs(W_clip), axis=1, keepdims=True) / float(W4A4_QMAX)
    w_scale = np.maximum(w_scale, 1e-12)
    W_q = fake_quantize_per_channel(W_clip, w_scale)
    a_scale = float(np.max(np.abs(x_s))) / float(W4A4_QMAX) if x_s.size else 0.0
    a_scale = max(a_scale, 1e-12)
    x_q = fake_quantize_per_tensor(x_s, a_scale)
    y_q = W_q @ x_q
    return y_ref, y_q


# ---------------------------------------------------------------------------
# Policy consumption (the integration surface)
# ---------------------------------------------------------------------------


def _pe_qat_policy_for(
    pe_qat_policy: Optional[dict],
    tensor_name: str,
) -> Optional[dict]:
    """Return the PE-QAT policy entry for `tensor_name`, or None.

    Two layouts are accepted:

    - Multi-tensor: ``{"tensors": {name: entry, ...}}`` -- exact match first,
      then substring match in either direction.  This is the layout a
      production orchestrator would write after training every dense layer.
    - Single-tensor flat: top-level ``lora_A`` / ``per_channel_smooth_s`` --
      implicit match (the policy itself is the entry).  This is the format
      ``save_pe_qat_policy`` produces for a single trained layer, so the
      existing demo output works without any wrapper.
    """
    if not pe_qat_policy:
        return None
    tensors = pe_qat_policy.get("tensors")
    if isinstance(tensors, dict):
        if tensor_name in tensors:
            entry = tensors[tensor_name]
            return entry if isinstance(entry, dict) else None
        for name, entry in tensors.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                continue
            if name and (name in tensor_name or tensor_name in name):
                return entry
        return None
    if "lora_A" in pe_qat_policy or "per_channel_smooth_s" in pe_qat_policy:
        return pe_qat_policy
    return None


def apply_pe_qat_to_weight(
    weight: np.ndarray,
    pe_qat_policy: dict,
    tensor_name: str,
) -> np.ndarray:
    """Apply a PE-QAT policy to a weight.

    The function is pure (no I/O, no mutation of the input) and runs in
    NumPy only.  The caller (e.g. ``quantize_v3.py``) is responsible for
    handing the returned adjusted weight to its quantizer.

    Steps:

    1. Look up the policy entry for ``tensor_name``.  Returns the original
       weight unchanged if no entry matches.
    2. If the entry carries LoRA factors, merge the low-rank delta into
       the weight: ``W' = W + (alpha / rank) * (B @ A)``.
    3. If the entry carries ``per_channel_smooth_s``, apply the
       SmoothQuant-style per-input-channel scaling: ``W'' = W' * s``.
       The clip threshold ``c`` is intentionally NOT applied here --
       it belongs at quantization time, where the per-output-channel
       maximum of ``W''`` is the natural reference scale.
    4. Return the adjusted weight as float32.

    Shape / dtype mismatches between the entry and ``weight`` cause the
    corresponding step to be skipped (with the weight passed through
    unchanged) rather than raising; the caller can decide whether the
    silent skip is acceptable.  The point is to make the integration
    fail-soft: a stale or partial policy never blocks quantization.
    """
    entry = _pe_qat_policy_for(pe_qat_policy, tensor_name)
    if entry is None:
        return weight.astype(np.float32, copy=False)

    if weight.ndim != 2:
        return weight.astype(np.float32, copy=False)
    out_features, in_features = weight.shape
    W = weight.astype(np.float32, copy=True)

    # 1. LoRA merge: W' = W + (alpha / rank) * (B @ A).
    lora_A = entry.get("lora_A")
    lora_B = entry.get("lora_B")
    if lora_A is not None and lora_B is not None:
        A = np.asarray(lora_A, dtype=np.float32)
        B = np.asarray(lora_B, dtype=np.float32)
        rank = int(entry.get("rank", A.shape[0] if A.ndim == 2 else 0))
        alpha = float(entry.get("alpha", float(rank)))
        if (
            A.ndim == 2
            and B.ndim == 2
            and rank > 0
            and A.shape == (rank, in_features)
            and B.shape == (out_features, rank)
            and alpha == alpha  # not NaN
        ):
            scaling = alpha / float(rank)
            W = W + scaling * (B @ A)
        # else: shape mismatch; skip LoRA merge (W' = W).

    # 2. Per-channel SmoothQuant-style scaling: W'' = W' * s.
    s = entry.get("per_channel_smooth_s")
    if s is not None:
        s_arr = np.asarray(s, dtype=np.float32)
        if s_arr.ndim == 1 and s_arr.shape[0] == in_features:
            W = W * s_arr[None, :]
        # else: shape mismatch; skip smooth scaling (W'' = W').
        # c (clip) is consumed at quantization time, not here.

    return W
