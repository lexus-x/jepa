"""Flow matching losses and metrics for VL-JEPA training.

Provides:
    - FlowMatchingLoss: MSE loss that accepts an external metric network
    - ConformalRegularizer: penalizes predictions outside conformal radius
    - TimestepSampler: Beta-distributed timestep sampling
    - Geodesic distance metrics for evaluation
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..flow.se3_utils import se3_logmap, se3_geodesic_distance, _se3_inverse


class FlowMatchingLoss(nn.Module):
    """Flow matching MSE loss with pluggable metric tensor.

    L = E_{t, x₀, x₁} ‖v_θ - u_t‖²_g

    The metric g is provided externally (from GeodesicFlowMatcher.metric_net)
    so there is exactly ONE metric network, not two disconnected copies.

    Args:
        metric_fn: Callable(task_embedding) → [B, 6] metric weights.
            If None, falls back to fixed identity metric [1,1,1,1,1,1].
        loss_type: "mse" or "huber".
        huber_delta: Huber loss delta.
    """

    def __init__(
        self,
        metric_fn: Optional[Callable[[Tensor], Tensor]] = None,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.metric_fn = metric_fn
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        task_embedding: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the flow matching loss.

        Args:
            predicted: [B, 6] predicted velocities in se(3).
            target: [B, 6] target velocities in se(3).
            task_embedding: [B, D] task features for metric conditioning.
                Required when metric_fn is set.
            mask: [B] optional sample mask.

        Returns:
            loss: Scalar loss value.
        """
        diff = predicted - target

        # Apply metric weighting
        if self.metric_fn is not None and task_embedding is not None:
            metric_weights = self.metric_fn(task_embedding)  # [B, 6]
        else:
            metric_weights = 1.0  # identity

        weighted_diff = diff * metric_weights

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


class ConformalRegularizer(nn.Module):
    """Penalize predictions whose conformal radius exceeds a threshold.

    During training, this encourages the model to produce predictions that
    stay within a reasonable conformal ball, preventing the model from
    learning degenerate solutions that require huge prediction sets.

    L_conf = λ * ReLU(r - r_max)²

    where r is the current conformal radius and r_max is the target max.

    Args:
        max_radius: Target maximum conformal radius.
        weight: Regularization weight λ.
    """

    def __init__(self, max_radius: float = 2.0, weight: float = 0.1) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.weight = weight

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        current_radius: float,
    ) -> Tensor:
        """Compute conformal regularization loss.

        Args:
            predicted: [B, 6] predicted velocities.
            target: [B, 6] target velocities.
            current_radius: Current conformal radius from the calibrator.

        Returns:
            loss: Scalar regularization loss.
        """
        if current_radius <= self.max_radius:
            return torch.tensor(0.0, device=predicted.device)

        excess = current_radius - self.max_radius
        return self.weight * excess ** 2


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
    """Compute geodesic distance metrics between predicted and true poses."""
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
    """Compute high-level action prediction metrics."""
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
