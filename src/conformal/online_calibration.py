"""Online adaptive conformal prediction using the Gibbs-Candès method.

Implements the online conformal prediction framework that adaptively adjusts
the miscoverage level α_t based on recent coverage feedback, with
exponentially decaying weights for historical scores.

Reference:
    Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift",
    NeurIPS 2021.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
from torch import Tensor

from .lie_scorer import LieScorer


class OnlineConformalCalibrator:
    """Gibbs-Candès adaptive conformal prediction for SE(3) actions.

    Maintains an online sequence of adjusted miscoverage levels α_t by
    tracking whether recent predictions cover the ground truth:

        α_{t+1} = α_t + η (α - 𝟙[s_t > q̂_t])

    where:
        - α is the target miscoverage rate
        - η is the learning rate
        - s_t is the nonconformity score at time t
        - q̂_t is the current quantile threshold
        - 𝟙[s_t > q̂_t] is 1 if the prediction missed

    Features:
        - Exponentially decaying weights for old scores (adaptation to drift)
        - Safety halt when the conformal radius grows too large
        - Online quantile estimation with streaming scores

    Args:
        alpha: Target miscoverage rate (e.g., 0.1 for 90% coverage).
        eta: Learning rate for α adaptation.
        decay: Exponential decay factor for old scores (0 < decay ≤ 1).
        window_size: Maximum number of recent scores to retain.
        safety_radius: Maximum allowed conformal radius before halting.
        min_scores: Minimum number of scores before making predictions.
        weight_rot: Rotation weight in geodesic norm.
        weight_trans: Translation weight in geodesic norm.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eta: float = 0.005,
        decay: float = 0.995,
        window_size: int = 1000,
        safety_radius: float = 10.0,
        min_scores: int = 30,
        weight_rot: float = 1.0,
        weight_trans: float = 1.0,
    ) -> None:
        assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
        assert 0 < decay <= 1, f"decay must be in (0, 1], got {decay}"
        assert 0 < eta, f"eta must be positive, got {eta}"

        self.alpha_target = alpha
        self.alpha_current = alpha
        self.eta = eta
        self.decay = decay
        self.window_size = window_size
        self.safety_radius = safety_radius
        self.min_scores = min_scores

        self.scorer = LieScorer(weight_rot=weight_rot, weight_trans=weight_trans)

        # Streaming state
        self._scores: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[int] = deque(maxlen=window_size)
        self._step: int = 0
        self._halted: bool = False
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, T_pred: Tensor, T_true: Tensor) -> dict:
        """Process a new prediction and update the calibrator.

        Args:
            T_pred: [1, 4, 4] or [B, 4, 4] predicted poses.
            T_true: [1, 4, 4] or [B, 4, 4] ground truth poses.

        Returns:
            info: Dictionary with:
                - "scores": nonconformity scores
                - "covered": whether each prediction covered the truth
                - "radius": current conformal radius
                - "alpha": current adjusted α
                - "halted": whether safety halt was triggered
        """
        if self._halted:
            return {
                "scores": None,
                "covered": None,
                "radius": float("inf"),
                "alpha": self.alpha_current,
                "halted": True,
                "halt_reason": self._halt_reason,
            }

        # Compute scores
        scores = self.scorer.score(T_pred, T_true)  # [B]

        # Current radius
        radius = self.current_radius()

        # Coverage check
        covered = scores <= radius  # [B]

        # Update streaming state
        for s in scores.detach().cpu().tolist():
            self._scores.append(s)
            self._timestamps.append(self._step)
            self._step += 1

        # Gibbs-Candès update: adjust α based on coverage
        # For each sample that missed, increase α (more lenient)
        # For each sample that covered, decrease α (more strict)
        for c in covered.detach().cpu().tolist():
            # 𝟙[missed] = 1 - 𝟙[covered]
            missed = 1.0 - float(c)
            self.alpha_current += self.eta * (self.alpha_target - missed)
            self.alpha_current = max(1e-4, min(0.5, self.alpha_current))

        # Safety check
        if radius > self.safety_radius:
            self._halted = True
            self._halt_reason = (
                f"Conformal radius {radius:.4f} exceeded safety threshold "
                f"{self.safety_radius:.4f}. Possible distribution shift."
            )

        return {
            "scores": scores.detach().cpu(),
            "covered": covered.detach().cpu(),
            "radius": radius,
            "alpha": self.alpha_current,
            "halted": self._halted,
        }

    def current_radius(self) -> float:
        """Compute the current conformal radius from streaming scores.

        Uses exponentially weighted quantile estimation.

        Returns:
            radius: Current prediction set radius.
        """
        if len(self._scores) < self.min_scores:
            return self.safety_radius  # Conservative default

        scores_tensor = torch.tensor(self._scores, dtype=torch.float32)
        timestamps_tensor = torch.tensor(self._timestamps, dtype=torch.float32)

        # Exponential decay weights: w_i = decay^(t_now - t_i)
        age = self._step - timestamps_tensor
        weights = self.decay ** age
        weights = weights / weights.sum()

        # Weighted quantile
        radius = self._weighted_quantile(scores_tensor, 1.0 - self.alpha_current, weights)
        return radius.item()

    def predict_with_radius(self, T_pred: Tensor) -> tuple[Tensor, float]:
        """Get prediction set with current adaptive radius.

        Args:
            T_pred: [B, 4, 4] predicted poses.

        Returns:
            samples: [B, K, 4, 4] prediction set samples.
            radius: Current conformal radius.
        """
        radius = self.current_radius()
        samples, mask = self.scorer.prediction_set(T_pred, radius)
        return samples, radius

    def is_covered(self, T_pred: Tensor, T_true: Tensor) -> Tensor:
        """Check if true poses fall within the current prediction set.

        Args:
            T_pred: [B, 4, 4] predicted poses.
            T_true: [B, 4, 4] true poses.

        Returns:
            covered: [B] boolean mask.
        """
        scores = self.scorer.score(T_pred, T_true)
        return scores <= self.current_radius()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        """Whether the safety halt has been triggered."""
        return self._halted

    @property
    def halt_reason(self) -> str:
        """Reason for safety halt, or empty string."""
        return self._halt_reason

    def recent_coverage(self, window: int = 100) -> Optional[float]:
        """Compute empirical coverage over the last `window` predictions.

        Returns:
            Coverage rate in [0, 1], or None if insufficient data.
        """
        if len(self._scores) < window:
            return None

        recent_scores = list(self._scores)[-window:]
        radius = self.current_radius()
        covered = sum(1 for s in recent_scores if s <= radius)
        return covered / window

    def state_dict(self) -> dict:
        """Serialize calibrator state for checkpointing."""
        return {
            "alpha_target": self.alpha_target,
            "alpha_current": self.alpha_current,
            "eta": self.eta,
            "decay": self.decay,
            "scores": list(self._scores),
            "timestamps": list(self._timestamps),
            "step": self._step,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore calibrator state from checkpoint."""
        self.alpha_target = state["alpha_target"]
        self.alpha_current = state["alpha_current"]
        self.eta = state["eta"]
        self.decay = state["decay"]
        self._scores = deque(state["scores"], maxlen=self.window_size)
        self._timestamps = deque(state["timestamps"], maxlen=self.window_size)
        self._step = state["step"]
        self._halted = state["halted"]
        self._halt_reason = state["halt_reason"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_quantile(scores: Tensor, q: float, weights: Tensor) -> Tensor:
        """Compute weighted quantile.

        Args:
            scores: [N] score values.
            q: Quantile level in (0, 1).
            weights: [N] normalized sample weights.

        Returns:
            Scalar quantile value.
        """
        sorted_indices = scores.argsort()
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cum_weights = sorted_weights.cumsum(dim=0)
        threshold = q
        idx = (cum_weights >= threshold).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return sorted_scores[-1]
        return sorted_scores[idx[0]]

    def reset(self) -> None:
        """Reset the calibrator to initial state."""
        self.alpha_current = self.alpha_target
        self._scores.clear()
        self._timestamps.clear()
        self._step = 0
        self._halted = False
        self._halt_reason = ""

    def __repr__(self) -> str:
        return (
            f"OnlineConformalCalibrator(α={self.alpha_target}, η={self.eta}, "
            f"decay={self.decay}, scores={len(self._scores)}, "
            f"α_curr={self.alpha_current:.4f}, halted={self._halted})"
        )
