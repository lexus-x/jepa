"""Random policy baseline.

Samples actions uniformly from SE(3) for comparison with learned policies.
Useful for verifying that the conformal layer works with any policy,
including degenerate ones.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import BasePolicy
from ..flow.se3_utils import se3_expmap


class RandomPolicy(BasePolicy):
    """Random SE(3) action sampler.

    Args:
        sigma_rot: Rotation noise scale.
        sigma_trans: Translation noise scale.
    """

    def __init__(
        self,
        sigma_rot: float = 0.3,
        sigma_trans: float = 0.1,
    ) -> None:
        self.sigma_rot = sigma_rot
        self.sigma_trans = sigma_trans

    def predict_action(
        self,
        observation: dict,
        instruction: str,
    ) -> Tensor:
        image = observation["image"]
        B = image.shape[0]
        device = image.device
        dtype = image.dtype

        xi = torch.zeros(B, 6, device=device, dtype=dtype)
        xi[:, :3] = self.sigma_rot * torch.randn(B, 3, device=device, dtype=dtype)
        xi[:, 3:] = self.sigma_trans * torch.randn(B, 3, device=device, dtype=dtype)

        identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        return se3_expmap(xi) @ identity

    def name(self) -> str:
        return "Random"
