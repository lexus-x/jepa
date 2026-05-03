"""π₀ (Pi-Zero) policy adapter.

Wraps the Physical Intelligence π₀ flow-matching VLA for use with the
conformal safety layer.  π₀ uses flow matching on normalized action
chunks which we convert to SE(3).

Requires: The π₀ model checkpoint and its dependencies.
    See: https://github.com/Physical-Intelligence/openpi
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import BasePolicy
from ..flow.se3_utils import se3_expmap


class PiZeroPolicy(BasePolicy):
    """Adapter for the π₀ VLA.

    π₀ predicts action chunks via flow matching.  We extract the first
    action, convert to SE(3), and return it.

    Args:
        checkpoint_path: Path to π₀ checkpoint.
        device: Target device.
    """

    def __init__(
        self,
        checkpoint_path: str = "physical-intelligence/pi0",
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            # π₀ uses its own loading convention
            from openpi.models import PiZeroModel
        except ImportError:
            raise ImportError(
                "openpi is required for π₀. "
                "Install via: pip install openpi  (or follow "
                "https://github.com/Physical-Intelligence/openpi)"
            )

        self._model = PiZeroModel.from_pretrained(
            self._checkpoint_path,
        ).to(self._device).eval()

    def predict_action(
        self,
        observation: dict,
        instruction: str,
    ) -> Tensor:
        self._ensure_loaded()
        assert self._model is not None

        image = observation["image"]
        if image.dim() == 5:
            image = image[:, :, 0]

        proprio = observation.get("proprioception")

        with torch.no_grad():
            # π₀ returns an action chunk [B, horizon, action_dim]
            action_chunk = self._model.predict(
                image=image,
                instruction=instruction,
                proprioception=proprio,
            )

        # Take the first action from the chunk
        action_first = action_chunk[:, 0]  # [B, action_dim]

        # Convert to SE(3)
        return self._action_to_se3(action_first)

    def _action_to_se3(self, action: Tensor) -> Tensor:
        """Convert π₀ action to SE(3).

        π₀ actions are typically 7D: [x, y, z, qx, qy, qz, qw] or
        [dx, dy, dz, ax, ay, az].  We handle both conventions.
        """
        B = action.shape[0]

        if action.shape[-1] >= 7:
            # Assume [x, y, z, qx, qy, qz, qw] → use position + axis-angle
            xi = torch.zeros(B, 6, device=action.device, dtype=action.dtype)
            xi[:, 3:] = action[:, :3]  # translation
            # Use first 3 of quaternion-ish as axis-angle proxy
            xi[:, :3] = action[:, 3:6] * 0.5  # small angle approx
        elif action.shape[-1] == 6:
            xi = action
        else:
            # Pad to 6D
            xi = torch.zeros(B, 6, device=action.device, dtype=action.dtype)
            xi[:, :action.shape[-1]] = action

        return se3_expmap(xi)

    def name(self) -> str:
        return "π₀"

    def to(self, device: torch.device) -> "PiZeroPolicy":
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
