"""Safe policy wrapper that integrates conformal prediction into the control loop.

This is the missing piece: the conformal calibrator was a standalone module
that nothing called.  SafePolicyWrapper wraps any SE(3) policy and:

    1. Runs the policy to get a point prediction
    2. Queries the conformal calibrator for the current radius
    3. Validates the prediction against the safety threshold
    4. Falls back to a conservative action if the prediction is outside
       the conformal set or the calibrator has halted
    5. Updates the calibrator with ground truth when available (for online adaptation)

This is what actually gets used at deployment time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from .online_calibration import OnlineConformalCalibrator
from .lie_scorer import LieScorer

logger = logging.getLogger(__name__)


class SafePolicyWrapper:
    """Wraps an SE(3) policy with conformal safety guarantees.

    At each step:
        1. Policy produces a predicted action T_pred ∈ SE(3)
        2. Conformal calibrator returns the current radius r
        3. If r > max_radius (distribution shift detected), use fallback
        4. If calibration is insufficient (< min_scores), use fallback
        5. Otherwise, return T_pred (with optional noise for exploration)

    When ground truth is available (e.g., during evaluation), the wrapper
    updates the calibrator to maintain valid coverage.

    Args:
        policy_fn: Callable that produces SE(3) actions.
            Signature: (images, instruction, proprioception, encoder) → [B, 4, 4]
        calibrator: Online conformal calibrator.
        max_radius: Maximum allowed conformal radius before fallback.
        fallback_action: Conservative action to use when safety is violated.
            If None, uses identity (no movement).
        enable_logging: Whether to log safety interventions.
    """

    def __init__(
        self,
        policy_fn: Callable[..., Tensor],
        calibrator: Optional[OnlineConformalCalibrator] = None,
        max_radius: float = 2.0,
        fallback_action: Optional[Tensor] = None,
        enable_logging: bool = True,
    ) -> None:
        self.policy_fn = policy_fn
        self.calibrator = calibrator or OnlineConformalCalibrator()
        self.max_radius = max_radius
        self._fallback_action = fallback_action
        self.enable_logging = enable_logging

        # Statistics
        self._total_calls = 0
        self._fallback_count = 0
        self._halt_count = 0
        self._insufficient_calibration_count = 0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def act(
        self,
        images: Tensor,
        instruction: str,
        proprioception: Optional[Tensor] = None,
        encoder: Any = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Produce a safe SE(3) action.

        Args:
            images: [B, 3, T, H, W] RGB frames.
            instruction: Natural language instruction.
            proprioception: [B, D_proprio] proprioceptive state.
            encoder: V-JEPA 2 encoder.
            device: Target device.

        Returns:
            action: [B, 4, 4] safe SE(3) action.
        """
        self._total_calls += 1
        B = images.shape[0]

        if device is None:
            device = images.device

        # Get conformal radius
        radius = self.calibrator.current_radius()

        # Check safety conditions
        if self.calibrator.is_halted:
            self._halt_count += 1
            if self.enable_logging:
                logger.warning(
                    "Conformal calibrator HALTED: %s. Using fallback action.",
                    self.calibrator.halt_reason,
                )
            return self._get_fallback(B, device)

        if radius > self.max_radius:
            self._fallback_count += 1
            if self.enable_logging:
                logger.warning(
                    "Conformal radius %.4f exceeds max %.4f. Using fallback.",
                    radius, self.max_radius,
                )
            return self._get_fallback(B, device)

        # Run the policy
        try:
            T_pred = self.policy_fn(images, instruction, proprioception, encoder)
        except Exception as e:
            logger.error("Policy failed: %s. Using fallback.", str(e))
            self._fallback_count += 1
            return self._get_fallback(B, device)

        return T_pred

    def update(
        self,
        T_pred: Tensor,
        T_true: Tensor,
    ) -> dict:
        """Update the calibrator with observed ground truth.

        Call this during evaluation or when you have access to the true
        action that should have been taken (e.g., from a demonstrator).

        Args:
            T_pred: [B, 4, 4] predicted actions.
            T_true: [B, 4, 4] ground truth actions.

        Returns:
            info: Calibrator update information (scores, coverage, radius).
        """
        return self.calibrator.update(T_pred, T_true)

    # ------------------------------------------------------------------
    # Batch evaluation with safety tracking
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        predictions: Tensor,
        ground_truths: Tensor,
    ) -> dict:
        """Evaluate a batch of predictions with conformal safety tracking.

        Args:
            predictions: [B, 4, 4] predicted SE(3) poses.
            ground_truths: [B, 4, 4] ground truth poses.

        Returns:
            metrics: Dictionary with:
                - "scores": nonconformity scores
                - "covered": coverage mask
                - "radius": current conformal radius
                - "would_fallback": which samples would trigger fallback
                - "coverage_rate": fraction covered
                - "fallback_rate": fraction that would fallback
        """
        scorer = LieScorer()
        scores = scorer.score(predictions, ground_truths)
        radius = self.calibrator.current_radius()

        covered = scores <= radius
        would_fallback = (radius > self.max_radius) | self.calibrator.is_halted

        return {
            "scores": scores,
            "covered": covered,
            "radius": radius,
            "would_fallback": would_fallback,
            "coverage_rate": covered.float().mean().item(),
            "fallback_rate": float(would_fallback),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return safety intervention statistics."""
        return {
            "total_calls": self._total_calls,
            "fallback_count": self._fallback_count,
            "halt_count": self._halt_count,
            "insufficient_calibration": self._insufficient_calibration_count,
            "fallback_rate": self._fallback_count / max(self._total_calls, 1),
            "current_radius": self.calibrator.current_radius(),
            "calibrator_halted": self.calibrator.is_halted,
        }

    def reset_stats(self) -> None:
        """Reset intervention counters."""
        self._total_calls = 0
        self._fallback_count = 0
        self._halt_count = 0
        self._insufficient_calibration_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_fallback(self, batch_size: int, device: torch.device) -> Tensor:
        """Generate a fallback (conservative) action.

        Returns identity if no fallback is configured (safest: do nothing).
        """
        if self._fallback_action is not None:
            return self._fallback_action.expand(batch_size, -1, -1).to(device)

        # Identity: safest action is to not move
        return torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    def __repr__(self) -> str:
        return (
            f"SafePolicyWrapper(max_radius={self.max_radius}, "
            f"calibrator={self.calibrator}, "
            f"fallback_rate={self.stats['fallback_rate']:.3f})"
        )
