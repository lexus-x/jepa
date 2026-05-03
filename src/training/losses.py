"""Flow matching losses and metrics for VL-JEPA training.

Provides:
    - FlowMatchingLoss: Weighted MSE loss for Riemannian flow matching
    - Timestep sampling from Beta distribution
    - Geodesic distance metric for evaluation

NOTE: The task-conditioned metric tensor is now part of GeodesicFlowMatcher
(TaskConditionedMetric). The loss here is a fallback for when the flow matcher's
own loss is not used directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..flow.se3_utils import se3_logmap, se3_geodesic_distance, _se3_inverse


class FlowMatchingLoss(nn.Module):
    """Flow matching MSE loss with optional metric tensor weighting.

    L = E_{t, x₀, x₁} ‖v_θ(x_t, t, z) - u_t‖²_g

    When metric_weights is None (default), uses the task-conditioned metric
    from GeodesicFlowMatcher instead. This class serves as a standalone
    fallback or for ablation studies.

    Args:
        weight_rot: Metric weight for rotation components (ω ∈ ℝ³).
        weight_trans: Metric weight for translation components (υ ∈ ℝ³).
        loss_type: Loss type - "mse" (default) or "huber".
        huber_delta: Huber loss delta.
    """

    def __init__(
        self,
        weight_rot: float = 1.0,
        weight_trans: float = 1.0,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.register_buffer(
            "metric_weights",
            torch.tensor([
                weight_rot, weight_rot, weight_rot,
                weight_trans, weight_trans, weight_trans,
            ]),
        )

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the flow matching loss with fixed metric weights.

        Args:
            predicted: [B, 6] predicted velocities in se(3).
            target: [B, 6] target velocities in se(3).
            mask: [B] optional sample mask.

        Returns:
            loss: Scalar loss value.
        """
        diff = predicted - target
        weighted_diff = diff * self.metric_weights.unsqueeze(0)

        if self.loss_type == "mse":
            per_sample = (weighted_diff ** 2).sum(dim=-1)
        elif self.loss_type == "huber":
            abs_diff = weighted_diff.abs()
            quadratic = 0.5 * abs_diff ** 2
            linear = self.huber_delta * (abs_diff - 0.5 * self.huber_delta)
            per_sample = torch.where(abs_diff <= self.huber_delta, quadratic, linear).sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if mask is not None:
            per_sample = per_sample * mask
            return per_sample.sum() / (mask.sum() + 1e-8)
        return per_sample.mean()


class TimestepSampler:
    """Timestep sampler using Beta(α, β) distribution.

    Args:
        alpha: Beta distribution α parameter.
        beta: Beta distribution β parameter.
        device: Default device.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.dist = torch.distributions.Beta(alpha, beta)
        self.device = device

    def sample(self, batch_size: int) -> Tensor:
        """Sample timesteps."""
        return self.dist.sample((batch_size,)).to(self.device)

    def to(self, device: torch.device) -> "TimestepSampler":
        self.device = device
        return self


def geodesic_distance_metric(T_pred: Tensor, T_true: Tensor) -> dict[str, Tensor]:
    """Compute geodesic distance metrics between predicted and true poses.

    Args:
        T_pred: [B, 4, 4] predicted SE(3) poses.
        T_true: [B, 4, 4] ground truth SE(3) poses.

    Returns:
        metrics: Dictionary with geodesic_total, rotation_error, translation_error.
    """
    total_dist = se3_geodesic_distance(T_pred, T_true)

    T_pred_inv = _se3_inverse(T_pred)
    delta = T_pred_inv @ T_true

    R_delta = delta[:, :3, :3]
    trace = R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    rot_error = cos_angle.acos()

    t_delta = delta[:, :3, 3]
    trans_error = t_delta.norm(dim=-1)

    return {
        "geodesic_total": total_dist,
        "rotation_error": rot_error,
        "translation_error": trans_error,
    }


def compute_action_metrics(
    predicted_actions: Tensor,
    ground_truth_actions: Tensor,
    threshold_rot: float = 0.1,
    threshold_trans: float = 0.01,
) -> dict[str, float]:
    """Compute high-level action prediction metrics.

    Args:
        predicted_actions: [B, 4, 4] predicted SE(3) actions.
        ground_truth_actions: [B, 4, 4] ground truth actions.
        threshold_rot: Rotation threshold for "close enough" (radians).
        threshold_trans: Translation threshold for "close enough" (meters).

    Returns:
        metrics: Dictionary with scalar metrics.
    """
    metrics = geodesic_distance_metric(predicted_actions, ground_truth_actions)

    rot_close = (metrics["rotation_error"] < threshold_rot).float().mean()
    trans_close = (metrics["translation_error"] < threshold_trans).float().mean()
    both_close = (
        (metrics["rotation_error"] < threshold_rot) &
        (metrics["translation_error"] < threshold_trans)
    ).float().mean()

    return {
        "geodesic_mean": metrics["geodesic_total"].mean().item(),
        "geodesic_std": metrics["geodesic_total"].std().item(),
        "rotation_error_mean": metrics["rotation_error"].mean().item(),
        "translation_error_mean": metrics["translation_error"].mean().item(),
        "rotation_within_threshold": rot_close.item(),
        "translation_within_threshold": trans_close.item(),
        "action_accuracy": both_close.item(),
    }
