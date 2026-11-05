"""Full-batch training of the penalised objective

    BCE(model) + lambda_res * sum_l c_l + lambda_out * 0.5 * (|W_out|^2 + |b_out|^2)

with Adam (no optimiser weight decay, linear warm-up and cosine decay of the learning
rate) followed by L-BFGS with strong Wolfe line search.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from resnet_theory.model import ResidualNet, head_cost, residual_cost


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def objective(model: ResidualNet, X: torch.Tensor, y: torch.Tensor, lambda_res: float, lambda_out: float):
    """Returns (penalised objective, data loss)."""
    loss = nn.functional.binary_cross_entropy_with_logits(model(X), y)
    return loss + lambda_res * residual_cost(model) + lambda_out * head_cost(model), loss


@torch.no_grad()
def evaluate(model: ResidualNet, X: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    logits = model(X)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    acc = ((torch.sigmoid(logits) >= 0.5).float() == y).float().mean()
    return {"loss": float(loss), "acc": float(acc)}


def learning_rate(step: int, total: int, base: float, warmup: int, min_ratio: float) -> float:
    """Linear warm-up over ``warmup`` steps, then cosine decay from ``base`` to ``min_ratio * base``."""
    warmup = min(warmup, total)
    if warmup > 0 and step <= warmup:
        return base * (step / warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (min_ratio + (1.0 - min_ratio) * cosine)


def train(
    model: ResidualNet,
    X: torch.Tensor,
    y: torch.Tensor,
    lambda_res: float,
    lambda_out: float,
    opt: dict[str, Any],
    log_every: int = 500,
) -> list[dict[str, Any]]:
    """Train in place and return a short log of the objective."""
    # Weight decay drives unused blocks to zero. Subnormal floats make CPU arithmetic
    # on such weights very slow, so they are flushed to zero.
    torch.set_flush_denormal(True)
    log: list[dict[str, Any]] = []

    def record(phase: str, step: int, total: torch.Tensor, loss: torch.Tensor) -> None:
        log.append({"phase": phase, "step": step, "objective": float(total.detach()), "loss": float(loss.detach()),
                    "residual_cost": float(residual_cost(model).detach())})

    adam_steps = int(opt["adam_steps"])
    if adam_steps > 0:
        adam = torch.optim.Adam(model.parameters(), lr=float(opt["adam_lr"]))
        for step in range(1, adam_steps + 1):
            lr = learning_rate(step, adam_steps, float(opt["adam_lr"]), int(opt["warmup_steps"]), float(opt["min_lr_ratio"]))
            for group in adam.param_groups:
                group["lr"] = lr
            adam.zero_grad()
            total, loss = objective(model, X, y, lambda_res, lambda_out)
            total.backward()
            if step == 1 or step % log_every == 0 or step == adam_steps:
                record("adam", step, total, loss)
            adam.step()

    lbfgs_steps = int(opt["lbfgs_steps"])
    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_steps,
            history_size=int(opt["lbfgs_history"]),
            tolerance_grad=float(opt["lbfgs_tolerance_grad"]),
            tolerance_change=float(opt["lbfgs_tolerance_change"]),
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad()
            total, loss = objective(model, X, y, lambda_res, lambda_out)
            total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % log_every == 0:
                record("lbfgs", evaluations, total, loss)
            return total

        lbfgs.step(closure)
        with torch.no_grad():
            total, loss = objective(model, X, y, lambda_res, lambda_out)
        record("lbfgs_end", evaluations, total, loss)
    return log
