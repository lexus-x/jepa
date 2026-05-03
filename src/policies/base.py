"""Abstract base class for VLA policies.

Any VLA that can produce SE(3) actions from observations can be wrapped
by the conformal safety layer.  This interface is intentionally minimal:
the conformal wrapper doesn't care how the policy works internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class BasePolicy(ABC):
    """Abstract base policy for SE(3) action prediction.

    Subclasses must implement ``predict_action`` which returns an SE(3)
    pose.  The conformal safety layer calls this method and wraps the
    result with coverage guarantees.

    The observation format is intentionally loose — each VLA has its own
    input spec.  The conformal layer only cares about the output (SE(3)).
    """

    @abstractmethod
    def predict_action(
        self,
        observation: dict,
        instruction: str,
    ) -> Tensor:
        """Predict an SE(3) action from an observation.

        Args:
            observation: Dict with at least:
                - "image": [B, C, H, W] or [B, C, T, H, W] RGB image(s)
                - "proprioception": [B, D_proprio] proprioceptive state (optional)
                - Any other VLA-specific keys
            instruction: Natural language task instruction.

        Returns:
            T_pred: [B, 4, 4] predicted SE(3) action(s).
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this policy."""
        ...

    def to(self, device: torch.device) -> "BasePolicy":
        """Move the policy to a device.  Override if needed."""
        return self

    def eval(self) -> "BasePolicy":
        """Set the policy to eval mode.  Override if needed."""
        return self

    @property
    def action_dim(self) -> int:
        """Dimensionality of the raw action space (before SE(3) conversion).

        Override if the VLA outputs something other than 6D (se(3)).
        """
        return 6

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
