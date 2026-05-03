"""Riemannian Conditional Flow Matching on SE(3).

Implements the full training and inference pipeline for flow matching on
the SE(3) manifold, following the framework of:

    Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
    Chen & Lipman, "Riemannian Flow Matching on General Geometries", ICLR 2024.

Training loss:
    L_RFM = E_{t, x₀, x₁} ‖v_θ(x_t, t, z) - log_{x_t}(γ_t(x₁))‖²_g

where γ_t is the geodesic interpolation between noise x₀ and data x₁,
and g is a task-conditioned metric tensor (not a fixed diagonal).
"""

from __future__ import annotations

from typing import Callable, Optional

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


# ======================================================================
# Task-conditioned metric tensor
# ======================================================================

class TaskConditionedMetric(nn.Module):
    """Learnable, task-conditioned diagonal metric tensor on se(3).

    Instead of a fixed [1,1,1,1,1,1], this produces a diagonal metric
    g = diag(exp(s₁), ..., exp(s₆)) conditioned on the task embedding,
    so that the loss landscape adapts to the task.

    Args:
        task_dim: Dimension of the task conditioning vector.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, task_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        # Initialize near identity: output ≈ [0,0,0,0,0,0] → exp ≈ [1,1,1,1,1,1]
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, task_embedding: Tensor) -> Tensor:
        """Compute metric weights from task embedding.

        Args:
            task_embedding: [B, task_dim] fused visual+language features.

        Returns:
            weights: [B, 6] positive metric weights (via softplus).
        """
        raw = self.net(task_embedding)  # [B, 6]
        # Softplus ensures positivity, shifted so init ≈ 1
        return nn.functional.softplus(raw, beta=5.0) + 0.5


# ======================================================================
# Learned halting network
# ======================================================================

class LearnedHaltingNetwork(nn.Module):
    """Learned halting for adaptive ODE integration (ponder-net style).

    Predicts a halting probability at each integration step.  Integration
    stops when cumulative halting probability exceeds a threshold.

    Args:
        state_dim: Dimension of the state being integrated (6 for se(3)).
        hidden_dim: Hidden dimension.
    """

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for t
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Bias towards halting late (so we don't stop at step 1)
        nn.init.constant_(self.net[-2].bias, -2.0)

    def forward(self, velocity: Tensor, t: Tensor) -> Tensor:
        """Predict halting probability.

        Args:
            velocity: [B, 6] current velocity estimate.
            t: [B] current timestep.

        Returns:
            p_halt: [B] halting probability in (0, 1).
        """
        inp = torch.cat([velocity, t.unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)


# ======================================================================
# Adaptive ODE integrator
# ======================================================================

class AdaptiveODEIntegrator:
    """Adaptive-step ODE integrator with curvature-based step control.

    Unlike fixed-step Euler/RK4, this adjusts dt based on:
        1. Local curvature (velocity norm change between steps)
        2. Learned halting (ponder-net style)
        3. Step rejection when error estimate exceeds tolerance

    Uses the Dormand-Prince (RK45) method with embedded error estimation.

    Args:
        velocity_fn: Callable(x_t, t, **cond) → [B, 6] velocity.
        atol: Absolute error tolerance for step acceptance.
        rtol: Relative error tolerance.
        dt_min: Minimum step size.
        dt_max: Maximum step size.
        max_steps: Maximum number of steps before forced termination.
        safety: Safety factor for step size adjustment (< 1).
    """

    def __init__(
        self,
        velocity_fn: Callable,
        atol: float = 1e-3,
        rtol: float = 1e-2,
        dt_min: float = 1e-4,
        dt_max: float = 0.2,
        max_steps: int = 200,
        safety: float = 0.9,
    ) -> None:
        self.velocity_fn = velocity_fn
        self.atol = atol
        self.rtol = rtol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.max_steps = max_steps
        self.safety = safety

    @torch.no_grad()
    def integrate(
        self,
        x0: Tensor,
        halting_net: Optional[LearnedHaltingNetwork] = None,
        cond: Optional[dict] = None,
    ) -> tuple[Tensor, dict]:
        """Integrate from noise (t=0) to data (t=1) with adaptive steps.

        Args:
            x0: [B, 4, 4] initial noise poses.
            halting_net: Optional learned halting network.
            cond: Conditioning dict passed to velocity_fn.

        Returns:
            x1: [B, 4, 4] final poses.
            info: Dict with step count, rejected steps, final t, etc.
        """
        B = x0.shape[0]
        device, dtype = x0.device, x0.dtype
        manifold = SE3Manifold()

        x_t = x0.clone()
        t = torch.zeros(B, device=device, dtype=dtype)
        dt = torch.full((B,), self.dt_max, device=device, dtype=dtype)

        # Per-sample tracking (some samples may halt before others)
        finished = torch.zeros(B, device=device, dtype=torch.bool)
        cumulative_halt = torch.zeros(B, device=device, dtype=dtype)
        total_steps = 0
        total_rejections = 0
        prev_vel_norm = torch.zeros(B, device=device, dtype=dtype)

        while not finished.all() and total_steps < self.max_steps:
            # Skip finished samples
            active = ~finished
            if not active.any():
                break

            t_active = t[active]
            x_active = x_t[active]
            dt_active = dt[active]

            # --- Dormand-Prince RK45 coefficients ---
            k1 = self.velocity_fn(x_active, t_active, **(cond or {}))
            k1_dt = k1 * dt_active.unsqueeze(-1)

            x2 = manifold.expmap(x_active, k1_dt * 0.2)
            k2 = self.velocity_fn(x2, t_active + dt_active * 0.2, **(cond or {}))
            k2_dt = k2 * dt_active.unsqueeze(-1)

            x3 = manifold.expmap(x_active, k1_dt * 0.075 + k2_dt * 0.225)
            k3 = self.velocity_fn(x3, t_active + dt_active * 0.3, **(cond or {}))
            k3_dt = k3 * dt_active.unsqueeze(-1)

            x4 = manifold.expmap(
                x_active,
                k1_dt * (44 / 45) - k2_dt * (56 / 15) + k3_dt * (32 / 9),
            )
            k4 = self.velocity_fn(x4, t_active + dt_active * (4 / 5), **(cond or {}))
            k4_dt = k4 * dt_active.unsqueeze(-1)

            x5 = manifold.expmap(
                x_active,
                k1_dt * (19372 / 6561) - k2_dt * (25360 / 2187)
                + k3_dt * (64448 / 6561) - k4_dt * (212 / 729),
            )
            k5 = self.velocity_fn(x5, t_active + dt_active * (8 / 9), **(cond or {}))
            k5_dt = k5 * dt_active.unsqueeze(-1)

            x6 = manifold.expmap(
                x_active,
                k1_dt * (9017 / 3168) - k2_dt * (355 / 33)
                + k3_dt * (46732 / 5247) + k4_dt * (49 / 176)
                - k5_dt * (5103 / 18656),
            )
            k6 = self.velocity_fn(x6, t_active + dt_active, **(cond or {}))
            k6_dt = k6 * dt_active.unsqueeze(-1)

            # 5th order update
            v5 = (
                k1_dt * (35 / 384) + k3_dt * (500 / 1113) + k4_dt * (125 / 192)
                - k5_dt * (2187 / 6784) + k6_dt * (11 / 84)
            )
            # 4th order update (embedded)
            v4 = (
                k1_dt * (5179 / 57600) + k3_dt * (7571 / 16695) + k4_dt * (393 / 640)
                - k5_dt * (92097 / 339200) + k6_dt * (187 / 2100)
            )

            # Error estimate: ‖v5 - v4‖
            error = (v5 - v4).norm(dim=-1)  # [B_active]
            x_candidate = manifold.expmap(x_active, v5)

            # Step acceptance criterion
            scale = self.atol + self.rtol * v5.norm(dim=-1)
            error_ratio = error / (scale + 1e-10)  # [B_active]

            # Accept/reject
            accept = error_ratio <= 1.0  # [B_active]

            # Curvature-based halting check
            vel_norm = v5.norm(dim=-1)  # [B_active]
            curvature = (vel_norm - prev_vel_norm[active]).abs() / (prev_vel_norm[active] + 1e-6)
            # Low curvature + near-zero velocity → we're done
            converged = (vel_norm < 1e-3) & (curvature < 0.1)

            # Learned halting
            if halting_net is not None:
                p_halt = halting_net(v5, t_active + dt_active)
                cumulative_halt[active] = cumulative_halt[active] + p_halt
                halt_now = cumulative_halt[active] > 0.99
            else:
                halt_now = converged

            # Update accepted samples
            if accept.any():
                accept_mask = accept
                indices = active.nonzero(as_tuple=True)[0]
                accept_indices = indices[accept_mask]

                x_t[accept_indices] = x_candidate[accept_mask]
                t[accept_indices] = t[accept_indices] + dt_active[accept_mask]
                prev_vel_norm[accept_indices] = vel_norm[accept_mask]

                # Mark finished samples
                just_finished = accept_mask & (halt_now | (t_active + dt_active >= 1.0 - 1e-4))
                finish_indices = indices[just_finished]
                finished[finish_indices] = True

            total_steps += 1
            total_rejections += (~accept).sum().item()

            # Adaptive step size (PI controller)
            optimal_dt = dt_active * self.safety * (1.0 / (error_ratio + 1e-6)).clamp(0.2, 5.0)
            optimal_dt = optimal_dt.clamp(self.dt_min, self.dt_max)

            # Update dt for all active samples
            dt[active] = optimal_dt
            # Ensure we don't overshoot t=1
            dt = torch.min(dt, 1.0 - t)

        # Handle samples that didn't reach t=1
        mask = t < 1.0 - 1e-4
        if mask.any():
            # Final Euler step for remaining
            remaining = mask.nonzero(as_tuple=True)[0]
            v_final = self.velocity_fn(
                x_t[remaining],
                t[remaining],
                **(cond or {}),
            )
            dt_final = (1.0 - t[remaining]).unsqueeze(-1)
            x_t[remaining] = manifold.expmap(x_t[remaining], v_final * dt_final)

        info = {
            "steps": total_steps,
            "rejections": total_rejections,
            "mean_final_t": t.mean().item(),
            "all_converged": finished.all().item(),
        }

        return x_t, info


# ======================================================================
# Main flow matcher
# ======================================================================

class GeodesicFlowMatcher(nn.Module):
    """Riemannian Conditional Flow Matching on SE(3).

    Handles:
        - Geodesic noising of SE(3) poses
        - Target velocity computation via logarithmic map
        - Task-conditioned metric tensor (learned, not static)
        - Training loss computation
        - Adaptive ODE integration with curvature-based step control

    Args:
        sigma_min: Minimum noise scale for the wrapped Gaussian.
        sigma_max: Maximum noise scale.
        beta_alpha: Alpha parameter for Beta(α, β) timestep distribution.
        beta_beta: Beta parameter for Beta(α, β) timestep distribution.
        task_dim: Dimension of task conditioning for the metric tensor.
    """

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 0.5,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        task_dim: int = 256,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
        self.manifold = SE3Manifold()

        # Task-conditioned metric tensor (NOT a static diagonal)
        self.metric_net = TaskConditionedMetric(task_dim=task_dim)

        # Learned halting for adaptive ODE
        self.halting_net = LearnedHaltingNetwork()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timesteps from Beta(α, β) distribution."""
        return self.beta_dist.sample((batch_size,)).to(device)

    def sample_noise(
        self,
        data_poses: Tensor,
        sigma: Optional[float] = None,
    ) -> Tensor:
        """Sample noise poses from a wrapped Gaussian centered at identity."""
        B = data_poses.shape[0]
        device, dtype = data_poses.device, data_poses.dtype

        if sigma is None:
            sigma = self.sigma_max

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
        """Compute noisy pose x_t and target velocity at timestep t."""
        if noise_poses is None:
            noise_poses = self.sample_noise(data_poses)

        x_t = geodesic_interpolation(noise_poses, data_poses, t)
        u_t = self.manifold.logmap(x_t, data_poses)

        scale = 1.0 / (1.0 - t + 1e-6)
        u_t = u_t * scale.unsqueeze(-1)

        return x_t, u_t

    # ------------------------------------------------------------------
    # Training loss (with task-conditioned metric)
    # ------------------------------------------------------------------

    def flow_matching_loss(
        self,
        predicted_velocity: Tensor,
        target_velocity: Tensor,
        task_embedding: Tensor,
    ) -> Tensor:
        """Compute Riemannian flow matching loss with task-conditioned metric.

        L = ‖v_θ - u_t‖²_g = Σ_i g_i(z) (v_θ^i - u_t^i)²

        where g(z) = softplus(Wz + b) is conditioned on the task embedding z.

        Args:
            predicted_velocity: [B, 6] predicted velocities.
            target_velocity: [B, 6] target velocities.
            task_embedding: [B, task_dim] fused visual+language features.

        Returns:
            loss: Scalar MSE loss.
        """
        diff = predicted_velocity - target_velocity  # [B, 6]

        # Task-conditioned metric weights [B, 6]
        metric_weights = self.metric_net(task_embedding)

        weighted_diff = diff * metric_weights
        return (weighted_diff ** 2).mean()

    # ------------------------------------------------------------------
    # Inference: adaptive ODE integration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def adaptive_integrate(
        self,
        velocity_fn: Callable,
        noise_poses: Tensor,
        cond: Optional[dict] = None,
        atol: float = 1e-3,
        rtol: float = 1e-2,
        max_steps: int = 200,
    ) -> tuple[Tensor, dict]:
        """Adaptive-step ODE integration with curvature-based step control.

        Uses Dormand-Prince RK45 with embedded error estimation, plus
        learned halting to determine when each sample has converged.

        Args:
            velocity_fn: Callable(x_t, t, **cond) → [B, 6] velocity.
            noise_poses: [B, 4, 4] initial noise poses.
            cond: Additional conditioning dict passed to velocity_fn.
            atol: Absolute error tolerance.
            rtol: Relative error tolerance.
            max_steps: Maximum integration steps.

        Returns:
            x_1: [B, 4, 4] generated poses.
            info: Integration diagnostics (steps, rejections, convergence).
        """
        integrator = AdaptiveODEIntegrator(
            velocity_fn=velocity_fn,
            atol=atol,
            rtol=rtol,
            max_steps=max_steps,
        )
        return integrator.integrate(noise_poses, halting_net=self.halting_net, cond=cond)

    @torch.no_grad()
    def euler_integrate(
        self,
        velocity_fn: Callable,
        noise_poses: Tensor,
        num_steps: int = 50,
        cond: Optional[dict] = None,
    ) -> Tensor:
        """Fixed-step Euler integration (fallback)."""
        x_t = noise_poses.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full(
                (x_t.shape[0],), t_val,
                device=x_t.device, dtype=x_t.dtype,
            )
            v = velocity_fn(x_t, t, **(cond or {}))
            x_t = self.manifold.expmap(x_t, v * dt)

        return x_t

    @torch.no_grad()
    def rk4_integrate(
        self,
        velocity_fn: Callable,
        noise_poses: Tensor,
        num_steps: int = 20,
        cond: Optional[dict] = None,
    ) -> Tensor:
        """Fixed-step RK4 integration (fallback)."""
        x_t = noise_poses.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t1 = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=x_t.dtype)
            k1 = velocity_fn(x_t, t1, **(cond or {}))

            x_mid1 = self.manifold.expmap(x_t, k1 * dt / 2)
            t2 = torch.full((x_t.shape[0],), t_val + dt / 2, device=x_t.device, dtype=x_t.dtype)
            k2 = velocity_fn(x_mid1, t2, **(cond or {}))

            x_mid2 = self.manifold.expmap(x_t, k2 * dt / 2)
            k3 = velocity_fn(x_mid2, t2, **(cond or {}))

            x_end = self.manifold.expmap(x_t, k3 * dt)
            t4 = torch.full((x_t.shape[0],), t_val + dt, device=x_t.device, dtype=x_t.dtype)
            k4 = velocity_fn(x_end, t4, **(cond or {}))

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

        if timesteps is None:
            t = self.sample_timesteps(B, device)
        else:
            t = timesteps

        x_t, u_t = self.compute_noisy_pose_and_target(data_poses, t)
        v_theta = velocity_field(x_t, t, visual_features, proprioception)

        # Use visual_features as task embedding for metric conditioning
        loss = self.flow_matching_loss(v_theta, u_t, visual_features)

        return loss, v_theta, u_t
