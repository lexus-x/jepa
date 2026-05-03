"""OpenVLA policy adapter.

Wraps the OpenVLA model (Kim et al., 2024) for use with the conformal
safety layer.  OpenVLA outputs 7D actions (dx, dy, dz, ax, ay, az, gripper)
which we convert to SE(3).

Requires: pip install openvla transformers
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .base import BasePolicy
from ..flow.se3_utils import se3_expmap


class OpenVLAPolicy(BasePolicy):
    """Adapter for the OpenVLA VLA.

    OpenVLA predicts 7D actions: [dx, dy, dz, ax, ay, az, gripper].
    We convert the first 6D to an SE(3) matrix via the exponential map.

    Args:
        model_name: HuggingFace model identifier.
            Default: "openvla/openvla-7b".
        device: Target device.
        dtype: Model dtype.
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for OpenVLA. "
                "Install via: pip install transformers"
            )

        self._processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device).eval()

    def predict_action(
        self,
        observation: dict,
        instruction: str,
    ) -> Tensor:
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        image = observation["image"]  # [B, C, H, W] or [B, C, T, H, W]
        if image.dim() == 5:
            image = image[:, :, 0]  # take first frame

        B = image.shape[0]
        device = image.device

        # Convert to PIL/numpy for the processor
        images_np = []
        for i in range(B):
            img = image[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            images_np.append(img)

        # Run OpenVLA inference
        actions_7d = []
        with torch.no_grad():
            for img in images_np:
                inputs = self._processor(
                    images=[img],
                    text=instruction,
                    return_tensors="pt",
                ).to(self._device, dtype=self._dtype)

                output = self._model.generate(**inputs, max_new_tokens=16)
                # OpenVLA returns action tokens that need decoding
                action = self._processor.decode(output[0], skip_special_tokens=True)
                # Parse the 7D action from the output string
                action_vec = self._parse_action(action)
                actions_7d.append(action_vec)

        actions_7d = torch.stack(actions_7d).to(device)  # [B, 7]

        # Convert to SE(3)
        return self._action_to_se3(actions_7d)

    def _parse_action(self, action_str: str) -> Tensor:
        """Parse a 7D action vector from OpenVLA output string."""
        try:
            values = [float(x) for x in action_str.strip().split()]
            if len(values) >= 7:
                return torch.tensor(values[:7], dtype=torch.float32)
        except (ValueError, IndexError):
            pass
        # Fallback: zero action
        return torch.zeros(7, dtype=torch.float32)

    def _action_to_se3(self, action_7d: Tensor) -> Tensor:
        """Convert 7D OpenVLA action to SE(3) matrix.

        Args:
            action_7d: [B, 7] = [dx, dy, dz, ax, ay, az, gripper]

        Returns:
            T: [B, 4, 4] SE(3) matrix.
        """
        xi = action_7d[:, :6]  # [B, 6]
        # Scale to reasonable range (OpenVLA actions are typically in [-1, 1])
        xi = xi * 0.1  # scale factor
        T = se3_expmap(xi)  # [B, 4, 4]
        return T

    def name(self) -> str:
        return f"OpenVLA({self._model_name})"

    def to(self, device: torch.device) -> "OpenVLAPolicy":
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
