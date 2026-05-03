"""Velocity field network for SE(3) flow matching.

Neural network that predicts tangent vectors in se(3) conditioned on:
    - Current SE(3) pose (encoded)
    - Timestep t ∈ (0, 1)
    - Visual features from V-JEPA 2
    - Proprioceptive state (joint angles, gripper state, etc.)

Architecture: MLP with sinusoidal time embedding and FiLM conditioning.
Also outputs a local curvature estimate used by the adaptive ODE integrator.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        t = t.unsqueeze(-1)
        args = t * self.freqs.unsqueeze(0)
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
        nn.init.ones_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply FiLM modulation."""
        return self.scale_proj(cond) * x + self.shift_proj(cond)


# ======================================================================
# SE(3) pose encoder
# ======================================================================

class SE3PoseEncoder(nn.Module):
    """Encode a 4×4 SE(3) matrix into a compact vector.

    Uses 6D continuous rotation representation (first two columns of R)
    plus translation, projected through an MLP.

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
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        rot_6d = R[:, :, :2].reshape(-1, 6)
        pose_vec = torch.cat([rot_6d, t], dim=-1)
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
        5. Auxiliary head: local curvature estimate for adaptive ODE

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

        # Conditioning fusion
        self.proprio_proj = nn.Linear(proprio_dim, 128)
        self.visual_proj = nn.Linear(visual_dim, 256)
        self.cond_fusion = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input: pose_emb + time_emb
        input_dim = pose_emb_dim + time_emb_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # MLP blocks with FiLM
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

        # Primary output: 6D se(3) velocity
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

        # Auxiliary output: local curvature estimate (scalar)
        # Used by the adaptive ODE integrator to adjust step sizes
        self.curvature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),  # Curvature is non-negative
        )

        # Initialize output head with small weights for stable training
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        nn.init.zeros_(self.curvature_head[-2].weight)
        nn.init.constant_(self.curvature_head[-2].bias, -1.0)  # Low initial curvature

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

        Returns:
            v: [B, 6] velocity in se(3).
        """
        h = self._encode(x_t, t, visual_features, proprioception)
        return self.output_head(h)

    def forward_with_curvature(
        self,
        x_t: Tensor,
        t: Tensor,
        visual_features: Tensor,
        proprioception: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict velocity AND local curvature estimate.

        Args:
            x_t: [B, 4, 4] current SE(3) poses.
            t: [B] timesteps in (0, 1).
            visual_features: [B, D_visual] visual conditioning.
            proprioception: [B, D_proprio] proprioceptive state.

        Returns:
            v: [B, 6] velocity in se(3).
            curvature: [B, 1] local curvature estimate (positive scalar).
        """
        h = self._encode(x_t, t, visual_features, proprioception)
        return self.output_head(h), self.curvature_head(h)

    def _encode(
        self,
        x_t: Tensor,
        t: Tensor,
        visual_features: Tensor,
        proprioception: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute shared hidden representation.

        Returns:
            h: [B, hidden_dim] hidden features.
        """
        B = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype

        pose_emb = self.pose_encoder(x_t)
        time_emb = self.time_embedding(t)

        visual_cond = self.visual_proj(visual_features)
        if proprioception is None:
            proprioception = torch.zeros(B, 7, device=device, dtype=dtype)
        proprio_cond = self.proprio_proj(proprioception)

        cond = self.cond_fusion(torch.cat([visual_cond, proprio_cond], dim=-1))
        h = self.input_proj(torch.cat([pose_emb, time_emb], dim=-1))

        for block, film, norm in zip(self.blocks, self.film_blocks, self.norms):
            h = h + block(h)
            h = norm(h)
            h = film(h, cond)

        return h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if not trainable_only or p.requires_grad
        )
