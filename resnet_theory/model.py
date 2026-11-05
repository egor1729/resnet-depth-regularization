"""Deep residual network with two-layer blocks.

    z_{l+1} = z_l + V_l phi(U_l z_l + a_l) + b_l,   l = 0, ..., L-1,

with phi(z) = z * sigmoid(z) (SiLU), followed by a linear read-out. There is no
normalisation and no branch multiplier. The cost of block l is

    c_l = 0.5 * (|U_l|_F^2 + |a_l|^2 + |V_l|_F^2 + |b_l|^2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def silu(z: torch.Tensor) -> torch.Tensor:
    return z * torch.sigmoid(z)


def _normal_init(layer: nn.Linear, std: float) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


class Block(nn.Module):
    def __init__(self, dim: int, width: int, init_scale: float):
        super().__init__()
        self.U = nn.Linear(dim, width)
        self.V = nn.Linear(width, dim)
        _normal_init(self.U, init_scale / dim**0.5)
        _normal_init(self.V, init_scale / width**0.5)

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        return self.V(silu(self.U(z)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.residual(z)


class ResidualNet(nn.Module):
    """Gaussian initialisation N(0, s^2 / fan_in) with zero biases.

    ``init`` has keys ``block_scale`` (s for U and V), ``head_scale`` (s for the read-out)
    and ``depth_scaling`` (if true, the block variance is further divided by L).
    """

    def __init__(self, dim: int, depth: int, width: int, init: dict, output_dim: int = 1):
        super().__init__()
        self.depth = depth
        depth_factor = max(1, depth) ** -0.5 if init.get("depth_scaling", False) else 1.0
        block_scale = float(init["block_scale"]) * depth_factor
        self.blocks = nn.ModuleList([Block(dim, width, block_scale) for _ in range(depth)])
        self.unembed = nn.Linear(dim, output_dim)
        _normal_init(self.unembed, float(init["head_scale"]) / dim**0.5)

    def features(self, x: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        """Final features, or the trajectory [L+1, N, d] of all intermediate states."""
        z = x
        states = [z]
        for block in self.blocks:
            z = block(z)
            states.append(z)
        return torch.stack(states) if return_trajectory else z

    def forward_range(self, start: int, end: int, z: torch.Tensor) -> torch.Tensor:
        """Apply blocks start, ..., end-1."""
        for block in self.blocks[start:end]:
            z = block(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.features(x))


def block_cost(block: Block) -> torch.Tensor:
    U, V = block.U, block.V
    cost = U.weight.square().sum() + U.bias.square().sum()
    cost = cost + V.weight.square().sum() + V.bias.square().sum()
    return 0.5 * cost


def residual_cost(model: ResidualNet) -> torch.Tensor:
    return torch.stack([block_cost(block) for block in model.blocks]).sum()


def head_cost(model: ResidualNet) -> torch.Tensor:
    head = model.unembed
    return 0.5 * (head.weight.square().sum() + head.bias.square().sum())


def load_state_dict(model: ResidualNet, state_dict: dict) -> None:
    """Load a checkpoint, also one written by the original experiment code (which stored
    an unused ``residual_scale`` buffer of value 1 per block)."""
    state = {k: v for k, v in state_dict.items() if not k.endswith("residual_scale")}
    model.load_state_dict(state)
