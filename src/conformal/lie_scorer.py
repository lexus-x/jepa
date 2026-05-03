"""Nonconformity scores on SE(3) for conformal prediction.

Implements geodesic nonconformity scores and prediction sets for
SE(3) action predictions, following the conformal prediction framework
adapted to Lie groups.

Reference:
    Shafer & Vovk, "A Tutorial on Conformal Prediction", 2008.
    Adapted to SE(3) using geodesic distance as the nonconformity measure.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..flow.se3_utils import se3_logmap, se3_geodesic_distance, _se3_inverse


class LieScorer:
    """Nonconformity scorer for SE(3) predictions.

    The nonconformity score measures how "surprising" a predicted pose is
    relative to a true pose, using the geodesic distance on SE(3):

        s(T_pred, T_true) = ‖log(T_pred⁻¹ T_true)‖

    This defines a geodesic ball prediction set:
        C_α = {T ∈ SE(3) : d_geo(T_pred, T) ≤ q̂_{1-α}}

    where q̂_{1-α} is the (1-α)-quantile of calibration scores.
    """

    def __init__(self, weight_rot: float = 1.0, weight_trans: float = 1.0) -> None:
        """Initialize the Lie scorer.

        Args:
            weight_rot: Weight for rotation components in the norm.
            weight_trans: Weight for translation components.
        """
        self.weights = torch.tensor(
            [weight_rot, weight_rot, weight_rot,
             weight_trans, weight_trans, weight_trans],
            dtype=torch.float32,
        )

    def score(self, T_pred: Tensor, T_true: Tensor) -> Tensor:
        """Compute nonconformity score: s(T_pred, T_true) = ‖log(T_pred⁻¹ T_true)‖_g.

        Args:
            T_pred: [B, 4, 4] predicted SE(3) poses.
            T_true: [B, 4, 4] ground truth SE(3) poses.

        Returns:
            scores: [B] nonconformity scores (non-negative).
        """
        # Relative transform: ΔT = T_pred⁻¹ @ T_true
        T_pred_inv = _se3_inverse(T_pred)
        delta = T_pred_inv @ T_true  # [B, 4, 4]

        # Log map: ξ = log(ΔT) ∈ se(3)
        xi = se3_logmap(delta)  # [B, 6]

        # Weighted norm
        device = xi.device
        weights = self.weights.to(device)
        scores = (xi * weights).norm(dim=-1)  # [B]

        return scores

    def score_batch(
        self,
        T_pred: Tensor,
        T_true: Tensor,
    ) -> Tensor:
        """Score a batch of predictions against ground truths.

        Handles mismatched batch dimensions by broadcasting.

        Args:
            T_pred: [B, 4, 4] or [B, N, 4, 4] predictions.
            T_true: [B, 4, 4] or [B, N, 4, 4] ground truths.

        Returns:
            scores: [B] or [B, N] nonconformity scores.
        """
        return self.score(T_pred, T_true)

    def prediction_set_radius(
        self,
        calibration_scores: Tensor,
        alpha: float,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the conformal prediction set radius (quantile).

        Args:
            calibration_scores: [N] scores from calibration set.
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage).
            weights: [N] optional sample weights for weighted quantile.

        Returns:
            radius: Scalar quantile value q̂_{1-α}.
        """
        if weights is not None:
            return self._weighted_quantile(calibration_scores, 1.0 - alpha, weights)
        else:
            return self._quantile(calibration_scores, 1.0 - alpha)

    def prediction_set(
        self,
        T_pred: Tensor,
        radius: float,
        num_samples: int = 1000,
    ) -> Tensor:
        """Generate a geodesic ball prediction set by sampling.

        C_α = {T ∈ SE(3) : d_geo(T_pred, T) ≤ radius}

        We approximate this by sampling from the wrapped Gaussian and filtering.

        Args:
            T_pred: [B, 4, 4] predicted poses.
            radius: Geodesic ball radius.
            num_samples: Number of samples to draw.

        Returns:
            samples: [B, K, 4, 4] poses in the prediction set.
                K may vary per batch element; padded with zeros.
        """
        from ..flow.se3_utils import wrapped_gaussian

        B = T_pred.shape[0]
        device = T_pred.device

        # Adaptive sigma based on radius
        sigma_trans = radius / 3.0
        sigma_rot = radius / 3.0

        samples = wrapped_gaussian(
            T_pred, sigma_trans=sigma_trans, sigma_rot=sigma_rot,
            num_samples=num_samples,
        )  # [B, num_samples, 4, 4]

        # Compute distances from prediction
        samples_flat = samples.reshape(B * num_samples, 4, 4)
        pred_expanded = T_pred.unsqueeze(1).expand(B, num_samples, 4, 4)
        pred_flat = pred_expanded.reshape(B * num_samples, 4, 4)

        distances = se3_geodesic_distance(pred_flat, samples_flat)  # [B * num_samples]
        distances = distances.reshape(B, num_samples)  # [B, num_samples]

        # Filter: keep samples within radius
        mask = distances <= radius  # [B, num_samples]

        return samples, mask

    @staticmethod
    def _quantile(scores: Tensor, q: float) -> Tensor:
        """Compute the q-quantile of scores.

        Uses the standard conformal quantile: ceiling of (n+1)q / n.
        """
        n = scores.shape[0]
        idx = int(torch.ceil(torch.tensor((n + 1) * q)).item())
        idx = min(max(idx, 1), n)
        sorted_scores, _ = scores.sort()
        return sorted_scores[idx - 1]

    @staticmethod
    def _weighted_quantile(scores: Tensor, q: float, weights: Tensor) -> Tensor:
        """Compute weighted quantile.

        Args:
            scores: [N] score values.
            q: Quantile level in (0, 1).
            weights: [N] sample weights.

        Returns:
            Scalar quantile value.
        """
        sorted_indices = scores.argsort()
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cum_weights = sorted_weights.cumsum(dim=0)
        threshold = q * cum_weights[-1]
        idx = (cum_weights >= threshold).nonzero(as_tuple=True)[0][0]
        return sorted_scores[idx]


class GeodesicBallPredictor:
    """Higher-level interface for conformal prediction on SE(3).

    Wraps calibration, scoring, and prediction set generation.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        weight_rot: float = 1.0,
        weight_trans: float = 1.0,
    ) -> None:
        """Initialize.

        Args:
            alpha: Target miscoverage rate.
            weight_rot: Rotation weight in geodesic norm.
            weight_trans: Translation weight in geodesic norm.
        """
        self.alpha = alpha
        self.scorer = LieScorer(weight_rot=weight_rot, weight_trans=weight_trans)
        self._calibration_scores: Optional[Tensor] = None
        self._radius: Optional[float] = None

    def calibrate(
        self,
        T_pred_cal: Tensor,
        T_true_cal: Tensor,
        weights: Optional[Tensor] = None,
    ) -> float:
        """Calibrate on held-out predictions.

        Args:
            T_pred_cal: [N, 4, 4] calibration predictions.
            T_true_cal: [N, 4, 4] calibration ground truths.
            weights: [N] optional sample weights.

        Returns:
            radius: Calibrated prediction set radius.
        """
        self._calibration_scores = self.scorer.score(T_pred_cal, T_true_cal)
        self._radius = self.scorer.prediction_set_radius(
            self._calibration_scores, self.alpha, weights,
        ).item()
        return self._radius

    @property
    def radius(self) -> float:
        """Return the calibrated radius."""
        if self._radius is None:
            raise RuntimeError("Must call calibrate() before accessing radius")
        return self._radius

    def predict(self, T_pred: Tensor) -> tuple[Tensor, float]:
        """Generate prediction set for new predictions.

        Args:
            T_pred: [B, 4, 4] predicted poses.

        Returns:
            samples: [B, K, 4, 4] prediction set samples.
            radius: The conformal radius used.
        """
        return self.scorer.prediction_set(T_pred, self.radius)

    def is_covered(self, T_pred: Tensor, T_true: Tensor) -> Tensor:
        """Check if true poses fall within the prediction set.

        Args:
            T_pred: [B, 4, 4] predicted poses.
            T_true: [B, 4, 4] true poses.

        Returns:
            covered: [B] boolean mask.
        """
        scores = self.scorer.score(T_pred, T_true)
        return scores <= self.radius
