"""V-JEPA 2 ViT-L/16 encoder wrapper.

Wraps the pretrained V-JEPA 2 model (300M params, 1024-dim output) for use as
a visual feature extractor in VL-JEPA.  Handles both single images and video
clips, applies ImageNet normalization, and supports optional fine-tuning of
the last N transformer layers.

Reference:
    Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding,
    Prediction and Planning", 2025.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


# ImageNet normalization constants
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class VJEPA2Encoder(nn.Module):
    """V-JEPA 2 ViT-L/16 encoder with optional fine-tuning of last N layers.

    Args:
        device: Target device for the model.
        dtype: Target dtype (default: float32).
        freeze: Whether to freeze all encoder parameters.
        unfreeze_last_n: Number of transformer layers to unfreeze from the end.
            Only effective when ``freeze=True``.
        temporal_patch_size: Temporal patch size of the encoder (default: 2).
        spatial_patch_size: Spatial patch size (default: 16).
        input_resolution: Expected spatial resolution (default: 256).
    """

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        temporal_patch_size: int = 2,
        spatial_patch_size: int = 16,
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.input_resolution = input_resolution

        # Number of spatial tokens per frame
        self.spatial_tokens_per_frame = (input_resolution // spatial_patch_size) ** 2  # 256

        # Load pretrained V-JEPA 2 ViT-L/16
        self.encoder = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_vit_large",
        )
        self.encoder = self.encoder.to(device=device, dtype=dtype)

        # ImageNet normalization (registered as buffer so it moves with the module)
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1, 1),
        )

        # Freeze / unfreeze logic
        if freeze:
            self._freeze_encoder(unfreeze_last_n)

        # Infer output dimension (ViT-L → 1024)
        self.output_dim: int = 1024

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_encoder(self, unfreeze_last_n: int) -> None:
        """Freeze all parameters, then optionally unfreeze the last N layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            # V-JEPA 2 ViT stores blocks in encoder.blocks (nn.ModuleList)
            blocks = getattr(self.encoder, "blocks", None)
            if blocks is None:
                # Fallback: search for the first ModuleList
                for module in self.encoder.modules():
                    if isinstance(module, nn.ModuleList):
                        blocks = module
                        break
            if blocks is not None and unfreeze_last_n <= len(blocks):
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True
                # Also unfreeze the final norm if present
                final_norm = getattr(self.encoder, "norm", None)
                if final_norm is not None:
                    for param in final_norm.parameters():
                        param.requires_grad = True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to RGB frames.

        Args:
            frames: [B, 3, T, H, W] in [0, 1].

        Returns:
            Normalized frames with same shape.
        """
        return (frames - self.mean) / self.std

    @staticmethod
    def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
        """Ensure input has shape [B, 3, T, H, W].

        Handles:
            - [3, H, W]          → [1, 3, 1, H, W]
            - [B, 3, H, W]       → [B, 3, 1, H, W]
            - [3, T, H, W]       → [1, 3, T, H, W]
            - [B, 3, T, H, W]    → unchanged
        """
        if x.dim() == 3:
            # Single image, no batch → [1, 3, 1, H, W]
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 4:
            # Could be [B,3,H,W] or [3,T,H,W]
            if x.shape[0] == 3:
                # Assume [3, T, H, W]
                x = x.unsqueeze(0)
            else:
                # Assume [B, 3, H, W] → add temporal dim
                x = x.unsqueeze(2)
        elif x.dim() != 5:
            raise ValueError(f"Expected 3D–5D input, got {x.dim()}D")
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode raw RGB frames into latent tokens.

        Args:
            frames: RGB frames in [0, 1] with shape compatible with
                [B, 3, T, H, W].  Single images are duplicated temporally.

        Returns:
            Latent tokens [B, N, 1024] where N = (T // temporal_patch_size) * spatial_tokens_per_frame.
        """
        frames = self._ensure_5d(frames)

        B, C, T, H, W = frames.shape

        # Duplicate single frame temporally if needed
        if T == 1:
            frames = frames.expand(B, C, self.temporal_patch_size, H, W)
            T = self.temporal_patch_size

        # Resize to expected resolution if needed
        if H != self.input_resolution or W != self.input_resolution:
            frames = F.interpolate(
                frames.reshape(B * C * T, 1, H, W),
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, C, T, self.input_resolution, self.input_resolution)

        # Normalize
        frames = self._normalize(frames)

        # V-JEPA 2 expects [B, C, T, H, W]
        tokens: torch.Tensor = self.encoder(frames)
        # tokens shape: [B, N, D] where N = (T // temporal_patch_size) * spatial_tokens_per_frame
        return tokens

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Forward pass (respects grad for fine-tuning).

        Args:
            frames: RGB frames [B, 3, T, H, W] in [0, 1].

        Returns:
            Latent tokens [B, N, 1024].
        """
        frames = self._ensure_5d(frames)

        B, C, T, H, W = frames.shape
        if T == 1:
            frames = frames.expand(B, C, self.temporal_patch_size, H, W)
            T = self.temporal_patch_size

        if H != self.input_resolution or W != self.input_resolution:
            frames = F.interpolate(
                frames.reshape(B * C * T, 1, H, W),
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, C, T, self.input_resolution, self.input_resolution)

        frames = self._normalize(frames)
        tokens: torch.Tensor = self.encoder(frames)
        return tokens

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the number of (optionally trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        total = self.num_parameters()
        trainable = self.num_parameters(trainable_only=True)
        return (
            f"output_dim={self.output_dim}, "
            f"temporal_patch={self.temporal_patch_size}, "
            f"spatial_patch={self.spatial_patch_size}, "
            f"params={total / 1e6:.1f}M (trainable={trainable / 1e6:.1f}M)"
        )
