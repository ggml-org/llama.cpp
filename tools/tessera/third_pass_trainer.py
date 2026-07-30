#!/usr/bin/env python3
"""Reusable MLX training step for Tessera's compression-aware third pass."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .third_pass_losses import LossWeights, joint_loss


class ThirdPassTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        weights: LossWeights = LossWeights(),
    ):
        self.model = model
        self.optimizer = optimizer
        self.weights = weights
        self.loss_and_grad = nn.value_and_grad(model, self._loss)
        self.step_number = 0

    def _loss(self, model, batch, teacher):
        student = model(batch)
        return joint_loss(
            student,
            teacher,
            batch["targets"],
            self.weights,
        )

    def step(self, batch, teacher) -> dict[str, float]:
        (loss, terms), gradients = self.loss_and_grad(
            self.model,
            batch,
            teacher,
        )
        self.optimizer.update(self.model, gradients)
        mx.eval(
            loss,
            terms,
            self.model.parameters(),
            self.optimizer.state,
        )
        self.step_number += 1
        return {
            "step": self.step_number,
            "loss": float(loss),
            **{name: float(value) for name, value in terms.items()},
        }

    def save_adapter(self, path: Path, metadata: dict[str, str] | None = None) -> None:
        tensors = {
            name: value
            for name, value in tree_flatten(self.model.trainable_parameters())
        }
        details = {
            "format": "mlx",
            "tessera.pass": "3",
            "tessera.loss_weights": str(asdict(self.weights)),
            **(metadata or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), tensors, metadata=details)
