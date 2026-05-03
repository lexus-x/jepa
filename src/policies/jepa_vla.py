"""JEPA-VLA policy adapter.

Wraps the JEPA-VLA model (Assran et al., 2025) for use with the conformal
safety layer.  JEPA-VLA uses V-JEPA 2 features with a flow-matching head
to predict SE(3) actions.

This adapter loads the JEPA-VLA checkpoint directly, using its own
architecture.  We don't reimplement the model — we just wrap it.

Requires: The JEPA-VLA codebase and checkpoints.
    See: https://github.com/facebookresearch/jepa-vla
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import BasePolicy


class JEPAPolicy(BasePolicy):
    """Adapter for the JEPA-VLA.

    Loads the JEPA-VLA model and exposes it through the standard
    policy interface.

    Args:
        checkpoint_path: Path to JEPA-VLA checkpoint.
        device: Target device.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/jepa_vla.pt",
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # JEPA-VLA uses its own model definition
        # This is a placeholder that loads the checkpoint
        state = torch.load(self._checkpoint_path, map_location=self._device)

        # The actual model architecture depends on the JEPA-VLA release
        # We expect the checkpoint to contain the model config
        if "config" in state:
            config = state["config"]
            # Build model from config (JEPA-VLA specific)
            from jepa_vla.model import build_model
            self._model = build_model(config).to(self._device)
        else:
            raise ValueError(
                f"JEPA-VLA checkpoint at {self._checkpoint_path} does not "
                f"contain a 'config' key.  Is this a valid JEPA-VLA checkpoint?"
            )

        self._model.load_state_dict(state["model"])
        self._model.eval()

    def predict_action(
        self,
        observation: dict,
        instruction: str,
    ) -> Tensor:
        self._ensure_loaded()
        assert self._model is not None

        image = observation["image"]
        proprio = observation.get("proprioception")

        with torch.no_grad():
            T_pred = self._model.predict_action(
                image=image,
                instruction=instruction,
                proprioception=proprio,
            )

        return T_pred  # [B, 4, 4]

    def name(self) -> str:
        return "JEPA-VLA"

    def to(self, device: torch.device) -> "JEPAPolicy":
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
