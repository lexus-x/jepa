"""Velocity field network for SE(3) flow matching.

Neural network that predicts tangent vectors in se(3) conditioned on:
    - Current SE(3) pose (encoded)
    - Timestep t ∈ (0, 1)
    - Visual features from V-JEPA 2
    - Proprioceptive state (joint angles, gripper state, etc.)

Architecture: MLP with sinusoidal time embedding and FiLM conditioning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .se3_utils import se3_logmap


# ======================================================================
# Sinusoidal time embedding
# ======================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar timesteps.

    Maps t ∈ ℝ → ℝ^{dim} using sin/cos at different frequencies.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        half_dim = dim // 2
        # Frequency bands: exp(-log(10000) * i / (half_dim - 1))
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """Encode timesteps.

        Args:
            t: [B] scalar timesteps.

        Returns:
            emb: [B, dim] sinusoidal embeddings.
        """
        # t: [B] → [B, 1]
        t = t.unsqueeze(-1)
        # [B, half_dim]
        args = t * self.freqs.unsqueeze(0)
        # [B, dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ======================================================================
# FiLM conditioning
# ======================================================================

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation.

    Applies scale and shift to a feature map conditioned on an external signal:
        y = γ(c) ⊙ x + β(c)

    Args:
        in_dim: Input feature dimension.
        cond_dim: Conditioning signal dimension.
    """

    def __init__(self, in_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, in_dim)
        self.shift_proj = nn.Linear(cond_dim, in_dim)
        # Initialize scale to 1 and shift to 0
        nn.init.ones_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply FiLM modulation.

        Args:
            x: [B, in_dim] features.
            cond: [B, cond_dim] conditioning.

        Returns:
            y: [B, in_dim] modulated features.
        """
        scale = self.scale_proj(cond)
        shift = self.shift_proj(cond)
        return scale * x + shift


# ======================================================================
# SE(3) pose encoder
# ======================================================================

class SE3PoseEncoder(nn.Module):
    """Encode a 4×4 SE(3) matrix into a compact vector.

    Extracts:
        - Rotation: 6D continuous rotation representation (first two columns)
        - Translation: 3D vector
    Total: 9D

    Args:
        output_dim: Output embedding dimension.
    """

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, T: Tensor) -> Tensor:
        """Encode SE(3) pose.

        Args:
            T: [B, 4, 4] SE(3) matrices.

        Returns:
            z: [B, output_dim] pose embeddings.
        """
        R = T[:, :3, :3]  # [B, 3, 3]
        t = T[:, :3, 3]  # [B, 3]

        # 6D rotation: first two columns of R
        rot_6d = R[:, :, :2].reshape(-1, 6)  # [B, 6]
        pose_vec = torch.cat([rot_6d, t], dim=-1)  # [B, 9]
        return self.mlp(pose_vec)


# ======================================================================
# Velocity field network
# ======================================================================

class VelocityField(nn.Module):
    """Predict se(3) velocity vectors conditioned on pose, time, vision, and proprioception.

    Architecture:
        1. Encode SE(3) pose → pose_emb
        2. Encode timestep → time_emb (sinusoidal)
        3. Fuse visual + proprioceptive features → cond_emb
        4. MLP with FiLM conditioning → 6D velocity in se(3)

    Args:
        visual_dim: Dimension of visual features (1024 for V-JEPA 2).
        proprio_dim: Dimension of proprioceptive state.
        pose_emb_dim: Dimension of pose embedding.
        time_emb_dim: Dimension of time embedding.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        proprio_dim: int = 7,
        pose_emb_dim: int = 64,
        time_emb_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pose_encoder = SE3PoseEncoder(pose_emb_dim)

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Conditioning fusion: visual + proprio → cond_dim
        self.proprio_proj = nn.Linear(proprio_dim, 128)
        self.visual_proj = nn.Linear(visual_dim, 256)
        self.cond_fusion = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input: pose_emb + time_emb
        input_dim = pose_emb_dim + time_emb_dim

        # Build MLP with FiLM conditioning
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        self.film_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
            self.film_blocks.append(FiLMBlock(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output head: 6D se(3) velocity
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

        # Initialize output head with small weights for stable training
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        visual_features: Tensor,
        proprioception: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict velocity at (x_t, t).

        Args:
            x_t: [B, 4, 4] current SE(3) poses.
            t: [B] timesteps in (0, 1).
            visual_features: [B, D_visual] visual conditioning.
            proprioception: [B, D_proprio] proprioceptive state.
                If None, uses zeros.

        Returns:
            v: [B, 6] velocity in se(3).
        """
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype

        # Encode pose
        pose_emb = self.pose_encoder(x_t)  # [B, pose_emb_dim]

        # Encode time
        time_emb = self.time_embedding(t)  # [B, time_emb_dim]

        # Build conditioning
        visual_cond = self.visual_proj(visual_features)  # [B, 256]

        if proprioception is None:
            proprioception = torch.zeros(B, 7, device=device, dtype=dtype)
        proprio_cond = self.proprio_proj(proprioception)  # [B, 128]

        cond = self.cond_fusion(torch.cat([visual_cond, proprio_cond], dim=-1))  # [B, hidden_dim]

        # Input
        h = self.input_proj(torch.cat([pose_emb, time_emb], dim=-1))  # [B, hidden_dim]

        # MLP blocks with FiLM
        for block, film, norm in zip(self.blocks, self.film_blocks, self.norms):
            h = h + block(h)  # Residual
            h = norm(h)
            h = film(h, cond)  # FiLM conditioning

        # Output
        v = self.output_head(h)  # [B, 6]

        return v

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if not trainable_only or p.requires_grad
        )
