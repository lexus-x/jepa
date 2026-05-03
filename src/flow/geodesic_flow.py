"""Riemannian Conditional Flow Matching on SE(3).

Implements the full training and inference pipeline for flow matching on
the SE(3) manifold, following the framework of:

    Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
    Chen & Lipman, "Riemannian Flow Matching on General Geometries", ICLR 2024.

Training loss:
    L_RFM = E_{t, x₀, x₁} ‖v_θ(x_t, t, z) - log_{x_t}(γ_t(x₁))‖²_g

where γ_t is the geodesic interpolation between noise x₀ and data x₁,
and g is an optional task-conditioned metric tensor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .se3_utils import (
    se3_expmap,
    se3_logmap,
    geodesic_interpolation,
    wrapped_gaussian,
    se3_geodesic_distance,
)
from .se3_manifold import SE3Manifold


class GeodesicFlowMatcher(nn.Module):
    """Riemannian Conditional Flow Matching on SE(3).

    Handles:
        - Geodesic noising of SE(3) poses
        - Target velocity computation via logarithmic map
        - Training loss computation
        - Inference via ODE integration (Euler / RK4)

    Args:
        sigma_min: Minimum noise scale for the wrapped Gaussian.
        sigma_max: Maximum noise scale.
        beta_alpha: Alpha parameter for Beta(α, β) timestep distribution.
        beta_beta: Beta parameter for Beta(α, β) timestep distribution.
        metric_weight_rot: Weight for rotation components in the metric tensor.
        metric_weight_trans: Weight for translation components.
    """

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 0.5,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        metric_weight_rot: float = 1.0,
        metric_weight_trans: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
        self.manifold = SE3Manifold()

        # Metric tensor weights (diagonal approximation)
        # g = diag(σ_rot, σ_rot, σ_rot, σ_trans, σ_trans, σ_trans)
        self.register_buffer(
            "metric_weights",
            torch.tensor([
                metric_weight_rot, metric_weight_rot, metric_weight_rot,
                metric_weight_trans, metric_weight_trans, metric_weight_trans,
            ]),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timesteps from Beta(α, β) distribution.

        Args:
            batch_size: Number of timesteps to sample.
            device: Target device.

        Returns:
            t: [B] timesteps in (0, 1).
        """
        return self.beta_dist.sample((batch_size,)).to(device)

    def sample_noise(
        self,
        data_poses: Tensor,
        sigma: Optional[float] = None,
    ) -> Tensor:
        """Sample noise poses from a wrapped Gaussian centered at identity.

        Args:
            data_poses: [B, 4, 4] data poses (for batch size / device inference).
            sigma: Noise scale (default: sigma_max).

        Returns:
            noise_poses: [B, 4, 4] noise poses on SE(3).
        """
        B = data_poses.shape[0]
        device, dtype = data_poses.device, data_poses.dtype

        if sigma is None:
            sigma = self.sigma_max

        # Sample noise in se(3) and exponentiate
        noise_vec = sigma * torch.randn(B, 6, device=device, dtype=dtype)
        identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        noise_poses = se3_expmap(noise_vec) @ identity
        return noise_poses

    # ------------------------------------------------------------------
    # Forward (noising + velocity target)
    # ------------------------------------------------------------------

    def compute_noisy_pose_and_target(
        self,
        data_poses: Tensor,
        t: Tensor,
        noise_poses: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute noisy pose x_t and target velocity at timestep t.

        The noisy pose is the geodesic interpolation:
            x_t = γ(t) = exp(t · log(x₁ x₀⁻¹)) x₀

        The target velocity is:
            u_t = log_{x_t}(γ'(t)) = log(x₁ x_t⁻¹) / (1 - t + ε)

        Args:
            data_poses: [B, 4, 4] clean data poses x₁.
            t: [B] timesteps in (0, 1).
            noise_poses: [B, 4, 4] noise poses x₀. If None, sampled automatically.

        Returns:
            x_t: [B, 4, 4] noisy poses.
            u_t: [B, 6] target tangent velocities in se(3).
        """
        if noise_poses is None:
            noise_poses = self.sample_noise(data_poses)

        # Geodesic interpolation: x_t = γ(t)
        x_t = geodesic_interpolation(noise_poses, data_poses, t)  # [B, 4, 4]

        # Target velocity: u_t = log_{x_t}(x₁) / (1 - t)
        # This is the velocity of the geodesic at time t
        u_t = self.manifold.logmap(x_t, data_poses)  # [B, 6]

        # Scale by 1/(1-t) to get the instantaneous velocity
        scale = 1.0 / (1.0 - t + 1e-6)  # [B]
        u_t = u_t * scale.unsqueeze(-1)

        return x_t, u_t

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def flow_matching_loss(
        self,
        predicted_velocity: Tensor,
        target_velocity: Tensor,
    ) -> Tensor:
        """Compute the Riemannian flow matching MSE loss.

        L = ‖v_θ - u_t‖²_g = Σ_i g_ii (v_θ^i - u_t^i)²

        Args:
            predicted_velocity: [B, 6] predicted velocities.
            target_velocity: [B, 6] target velocities.

        Returns:
            loss: Scalar MSE loss.
        """
        diff = predicted_velocity - target_velocity  # [B, 6]
        # Weighted MSE using metric tensor
        weighted_diff = diff * self.metric_weights.unsqueeze(0)
        return (weighted_diff ** 2).mean()

    # ------------------------------------------------------------------
    # Inference: ODE integration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_integrate(
        self,
        velocity_fn,
        noise_poses: Tensor,
        num_steps: int = 50,
        cond: Optional[dict] = None,
    ) -> Tensor:
        """Euler integration of the learned ODE from noise to data.

        dx/dt = v_θ(x, t, z),  x(0) = noise,  x(1) = data

        Args:
            velocity_fn: Callable(x_t, t, **cond) → [B, 6] velocity.
            noise_poses: [B, 4, 4] initial noise poses.
            num_steps: Number of integration steps.
            cond: Additional conditioning dict passed to velocity_fn.

        Returns:
            x_1: [B, 4, 4] generated poses.
        """
        x_t = noise_poses.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full(
                (x_t.shape[0],), t_val,
                device=x_t.device, dtype=x_t.dtype,
            )
            v = velocity_fn(x_t, t, **(cond or {}))  # [B, 6]

            # Update: x_{t+dt} = exp(dt * v) @ x_t
            x_t = self.manifold.expmap(x_t, v * dt)

        return x_t

    @torch.no_grad()
    def rk4_integrate(
        self,
        velocity_fn,
        noise_poses: Tensor,
        num_steps: int = 20,
        cond: Optional[dict] = None,
    ) -> Tensor:
        """RK4 integration of the learned ODE.

        More accurate than Euler with fewer steps.

        Args:
            velocity_fn: Callable(x_t, t, **cond) → [B, 6] velocity.
            noise_poses: [B, 4, 4] initial noise poses.
            num_steps: Number of integration steps.
            cond: Additional conditioning dict passed to velocity_fn.

        Returns:
            x_1: [B, 4, 4] generated poses.
        """
        x_t = noise_poses.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt

            # k1
            t1 = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=x_t.dtype)
            k1 = velocity_fn(x_t, t1, **(cond or {}))

            # k2
            x_mid1 = self.manifold.expmap(x_t, k1 * dt / 2)
            t2 = torch.full((x_t.shape[0],), t_val + dt / 2, device=x_t.device, dtype=x_t.dtype)
            k2 = velocity_fn(x_mid1, t2, **(cond or {}))

            # k3
            x_mid2 = self.manifold.expmap(x_t, k2 * dt / 2)
            k3 = velocity_fn(x_mid2, t2, **(cond or {}))

            # k4
            x_end = self.manifold.expmap(x_t, k3 * dt)
            t4 = torch.full((x_t.shape[0],), t_val + dt, device=x_t.device, dtype=x_t.dtype)
            k4 = velocity_fn(x_end, t4, **(cond or {}))

            # Combined update
            v_combined = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            x_t = self.manifold.expmap(x_t, v_combined * dt)

        return x_t

    # ------------------------------------------------------------------
    # Full forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        data_poses: Tensor,
        velocity_field: nn.Module,
        visual_features: Tensor,
        proprioception: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full training forward pass.

        1. Sample timesteps and noise
        2. Compute noisy poses and target velocities
        3. Predict velocities
        4. Compute loss

        Args:
            data_poses: [B, 4, 4] ground truth SE(3) poses.
            velocity_field: VelocityField network.
            visual_features: [B, D_visual] visual conditioning.
            proprioception: [B, D_proprio] proprioceptive state (optional).
            timesteps: [B] pre-sampled timesteps (optional).

        Returns:
            loss: Scalar training loss.
            predicted_velocity: [B, 6] predicted velocities.
            target_velocity: [B, 6] target velocities.
        """
        B = data_poses.shape[0]
        device = data_poses.device

        # Sample timesteps
        if timesteps is None:
            t = self.sample_timesteps(B, device)
        else:
            t = timesteps

        # Compute noisy pose and target velocity
        x_t, u_t = self.compute_noisy_pose_and_target(data_poses, t)

        # Predict velocity
        v_theta = velocity_field(x_t, t, visual_features, proprioception)

        # Loss
        loss = self.flow_matching_loss(v_theta, u_t)

        return loss, v_theta, u_t
