"""Comparison with SAFE (scalar failure detection) baseline.

SAFE (Zhao et al., 2024) detects policy failures using a scalar
confidence score.  Our approach uses SE(3) conformal prediction sets
which provide:

    1. Geometric information: WHERE the failure mode is (rotation vs translation)
    2. Tighter bounds: geodesic ball vs scalar threshold
    3. Distribution-free guarantees: no assumptions on failure distribution

This module runs both methods on the same trajectories and compares:
    - Coverage rate (both should be ≥ 1-α)
    - Prediction set size (ours should be tighter)
    - Failure localization (ours provides SE(3) geometry)
    - Adaptation speed (Gibbs-Candès vs fixed threshold)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..conformal.lie_scorer import LieScorer
from ..conformal.online_calibration import OnlineConformalCalibrator

logger = logging.getLogger(__name__)


class ScalarFailureDetector(nn.Module):
    """SAFE-style scalar failure detector.

    Trains a small MLP to predict a failure probability from the same
    features the conformal method uses.  At test time, flags failures
    when the predicted probability exceeds a calibrated threshold.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Predict failure probability.

        Args:
            features: [B, D] input features (e.g., predicted action).

        Returns:
            p_fail: [B] failure probability in (0, 1).
        """
        return self.net(features).squeeze(-1)


class SAFEComparison:
    """Compare SE(3) conformal prediction vs SAFE scalar detection.

    Runs both methods on the same evaluation data and produces
    comparative metrics.

    Args:
        alpha: Target miscoverage rate for conformal.
        safe_threshold: Scalar threshold for SAFE detector.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        safe_threshold: float = 0.5,
    ) -> None:
        self.alpha = alpha
        self.safe_threshold = safe_threshold

        # Conformal components
        self.conformal = OnlineConformalCalibrator(alpha=alpha)
        self.scorer = LieScorer()

        # SAFE detector
        self.safe_detector = ScalarFailureDetector()
        self._safe_calibrated = False
        self._safe_threshold = safe_threshold

    def calibrate(
        self,
        T_pred_cal: Tensor,
        T_true_cal: Tensor,
        failure_labels: Optional[Tensor] = None,
    ) -> dict:
        """Calibrate both methods on held-out data.

        Args:
            T_pred_cal: [N, 4, 4] calibration predictions.
            T_true_cal: [N, 4, 4] calibration ground truths.
            failure_labels: [N] binary failure labels (1 = failure).
                If None, derived from geodesic distance.

        Returns:
            cal_info: Calibration results.
        """
        # Conformal calibration
        conformal_scores = self.scorer.score(T_pred_cal, T_true_cal)

        # SAFE calibration: fit threshold from scores
        if failure_labels is None:
            # Derive failure labels from geodesic distance
            failure_labels = (conformal_scores > conformal_scores.median()).float()

        # Train SAFE detector
        xi_pred = self._extract_features(T_pred_cal)
        self.safe_detector.train()
        optimizer = torch.optim.Adam(self.safe_detector.parameters(), lr=1e-3)
        for _ in range(100):
            p_fail = self.safe_detector(xi_pred)
            loss = nn.functional.binary_cross_entropy(p_fail, failure_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.safe_detector.eval()

        # Calibrate SAFE threshold for desired coverage
        with torch.no_grad():
            p_fail_cal = self.safe_detector(xi_pred)
        # Find threshold that gives (1-alpha) coverage
        sorted_probs, _ = p_fail_cal.sort()
        idx = int((1 - self.alpha) * len(sorted_probs))
        self._safe_threshold = sorted_probs[min(idx, len(sorted_probs) - 1)].item()
        self._safe_calibrated = True

        return {
            "conformal_scores": conformal_scores,
            "safe_threshold": self._safe_threshold,
            "safe_loss": loss.item(),
        }

    def compare(
        self,
        T_pred: Tensor,
        T_true: Tensor,
    ) -> dict:
        """Compare both methods on test data.

        Args:
            T_pred: [B, 4, 4] predicted actions.
            T_true: [B, 4, 4] ground truth actions.

        Returns:
            comparison: Dict with:
                - "conformal_coverage": coverage rate of conformal sets
                - "safe_coverage": coverage rate of SAFE detector
                - "conformal_radius": conformal prediction set radius
                - "conformal_scores": per-sample nonconformity scores
                - "safe_probs": per-sample SAFE failure probabilities
                - "conformal_set_size": relative size of SE(3) prediction set
                - "safe_set_size": scalar threshold (no geometry)
                - "localization": per-axis error breakdown (conformal only)
        """
        B = T_pred.shape[0]

        # --- Conformal ---
        conformal_scores = self.scorer.score(T_pred, T_true)
        conformal_radius = self.conformal.current_radius()
        conformal_covered = conformal_scores <= conformal_radius

        # Update conformal calibrator
        self.conformal.update(T_pred, T_true)

        # Per-axis breakdown (rotation vs translation)
        T_pred_inv = self.scorer._inverse(T_pred)
        delta = T_pred_inv @ T_true
        from ..flow.se3_utils import se3_logmap
        xi = se3_logmap(delta)  # [B, 6]
        rot_error = xi[:, :3].norm(dim=-1)  # [B]
        trans_error = xi[:, 3:].norm(dim=-1)  # [B]

        # --- SAFE ---
        xi_pred = self._extract_features(T_pred)
        with torch.no_grad():
            safe_probs = self.safe_detector(xi_pred)
        safe_covered = safe_probs <= self._safe_threshold

        # --- Conformal set size (volume of geodesic ball in SE(3)) ---
        # Vol(B_r) in SE(3) ∝ r^6 (6D manifold)
        conformal_volume = conformal_radius ** 6

        return {
            "conformal_coverage": conformal_covered.float().mean().item(),
            "safe_coverage": safe_covered.float().mean().item(),
            "conformal_radius": conformal_radius,
            "conformal_scores": conformal_scores,
            "safe_probs": safe_probs,
            "conformal_set_size": conformal_volume,
            "safe_set_size": self._safe_threshold,  # scalar, no geometry
            "localization": {
                "rotation_error": rot_error,
                "translation_error": trans_error,
                "rotation_within_radius": (rot_error <= conformal_radius).float().mean().item(),
                "translation_within_radius": (trans_error <= conformal_radius).float().mean().item(),
            },
        }

    def _extract_features(self, T: Tensor) -> Tensor:
        """Extract features from SE(3) matrices for SAFE detector.

        Uses the se(3) log coordinates as features.
        """
        from ..flow.se3_utils import se3_logmap
        return se3_logmap(T)  # [B, 6]

    def summary(self, comparisons: list[dict]) -> dict:
        """Aggregate multiple comparison results.

        Args:
            comparisons: List of compare() outputs.

        Returns:
            summary: Aggregated metrics.
        """
        import numpy as np

        conf_cov = [c["conformal_coverage"] for c in comparisons]
        safe_cov = [c["safe_coverage"] for c in comparisons]
        conf_rad = [c["conformal_radius"] for c in comparisons]

        return {
            "conformal_mean_coverage": np.mean(conf_cov),
            "safe_mean_coverage": np.mean(safe_cov),
            "conformal_mean_radius": np.mean(conf_rad),
            "conformal_coverage_std": np.std(conf_cov),
            "safe_coverage_std": np.std(safe_cov),
            "advantage_coverage": np.mean(conf_cov) - np.mean(safe_cov),
            "conformal_provides_geometry": True,
            "safe_provides_geometry": False,
        }
