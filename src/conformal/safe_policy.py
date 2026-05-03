"""Safe policy wrapper: conformal safety for any VLA.

This is the main entry point for the framework.  Wraps any ``BasePolicy``
with conformal prediction guarantees on SE(3):

    1. Policy predicts T_pred ∈ SE(3)
    2. Conformal calibrator returns radius r
    3. If r > threshold or calibrator halted → fallback action
    4. Otherwise → return T_pred (with coverage guarantee)

The wrapper also tracks statistics for benchmarking: coverage rate,
intervention rate, radius evolution.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from ..policies.base import BasePolicy
from .online_calibration import OnlineConformalCalibrator
from .lie_scorer import LieScorer

logger = logging.getLogger(__name__)


class SafePolicyWrapper:
    """Conformal safety wrapper for SE(3) policies.

    Wraps any ``BasePolicy`` with:
        - Online conformal calibration (Gibbs-Candès)
        - Geodesic ball prediction sets on SE(3)
        - Safety halt + fallback when distribution shifts
        - Statistics tracking for benchmarking

    Args:
        policy: Any VLA that implements ``predict_action``.
        calibrator: Online conformal calibrator.  If None, creates a default.
        max_radius: Maximum allowed conformal radius before fallback.
        fallback_action: Conservative fallback.  If None, uses identity (no move).
        enable_logging: Whether to log safety interventions.
    """

    def __init__(
        self,
        policy: BasePolicy,
        calibrator: Optional[OnlineConformalCalibrator] = None,
        max_radius: float = 2.0,
        fallback_action: Optional[Tensor] = None,
        enable_logging: bool = True,
    ) -> None:
        self.policy = policy
        self.calibrator = calibrator or OnlineConformalCalibrator()
        self.max_radius = max_radius
        self._fallback_action = fallback_action
        self.enable_logging = enable_logging

        # Scorer for evaluation
        self._scorer = LieScorer()

        # Statistics
        self._total_calls = 0
        self._fallback_count = 0
        self._halt_count = 0
        self._radius_history: list[float] = []
        self._coverage_history: list[bool] = []

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def act(
        self,
        observation: dict,
        instruction: str,
    ) -> tuple[Tensor, dict]:
        """Produce a safe SE(3) action.

        Args:
            observation: Dict with at least "image" key.
            instruction: Natural language task instruction.

        Returns:
            action: [B, 4, 4] safe SE(3) action.
            info: Dict with:
                - "radius": current conformal radius
                - "fallback": whether fallback was used
                - "halted": whether calibrator is halted
                - "raw_action": the policy's original prediction (if different)
        """
        self._total_calls += 1
        radius = self.calibrator.current_radius()
        self._radius_history.append(radius)

        # Safety checks
        if self.calibrator.is_halted:
            self._halt_count += 1
            if self.enable_logging:
                logger.warning("HALTED: %s", self.calibrator.halt_reason)
            return self._get_fallback(observation), {
                "radius": radius, "fallback": True, "halted": True,
                "raw_action": None,
            }

        if radius > self.max_radius:
            self._fallback_count += 1
            if self.enable_logging:
                logger.warning("Radius %.4f > max %.4f", radius, self.max_radius)
            return self._get_fallback(observation), {
                "radius": radius, "fallback": True, "halted": False,
                "raw_action": None,
            }

        # Run policy
        try:
            T_pred = self.policy.predict_action(observation, instruction)
        except Exception as e:
            logger.error("Policy failed: %s", str(e))
            self._fallback_count += 1
            return self._get_fallback(observation), {
                "radius": radius, "fallback": True, "halted": False,
                "raw_action": None,
            }

        return T_pred, {
            "radius": radius, "fallback": False, "halted": False,
            "raw_action": T_pred,
        }

    def update(
        self,
        T_pred: Tensor,
        T_true: Tensor,
    ) -> dict:
        """Update the calibrator with observed ground truth.

        Call this during evaluation with the true action that should have
        been taken (e.g., from a demonstrator or simulator).

        Args:
            T_pred: [B, 4, 4] predicted actions.
            T_true: [B, 4, 4] ground truth actions.

        Returns:
            info: Calibrator update info (scores, coverage, radius).
        """
        result = self.calibrator.update(T_pred, T_true)
        # Track coverage
        if result["covered"] is not None:
            for c in result["covered"].tolist():
                self._coverage_history.append(bool(c))
        return result

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def evaluate_trajectory(
        self,
        trajectory: list[tuple[dict, str, Tensor]],
    ) -> dict:
        """Evaluate a full trajectory with conformal tracking.

        Args:
            trajectory: List of (observation, instruction, ground_truth_action).

        Returns:
            results: Dict with per-step and aggregate metrics.
        """
        predictions = []
        ground_truths = []
        actions_taken = []
        radii = []
        fallbacks = []

        for obs, instr, gt_action in trajectory:
            T_pred, info = self.act(obs, instr)
            predictions.append(T_pred)
            ground_truths.append(gt_action)
            actions_taken.append(T_pred if not info["fallback"] else self._get_fallback(obs))
            radii.append(info["radius"])
            fallbacks.append(info["fallback"])

            # Update calibrator with ground truth
            self.update(T_pred, gt_action.unsqueeze(0))

        predictions = torch.cat(predictions, dim=0)
        ground_truths = torch.stack(ground_truths)
        actions_taken = torch.cat(actions_taken, dim=0)

        # Compute scores
        scores = self._scorer.score(predictions, ground_truths)

        return {
            "scores": scores,
            "radii": radii,
            "fallbacks": fallbacks,
            "mean_score": scores.mean().item(),
            "mean_radius": sum(radii) / len(radii),
            "fallback_rate": sum(fallbacks) / len(fallbacks),
            "coverage_rate": sum(self._coverage_history[-len(trajectory):]) / len(trajectory),
        }

    @property
    def stats(self) -> dict:
        """Return aggregate statistics."""
        return {
            "total_calls": self._total_calls,
            "fallback_count": self._fallback_count,
            "halt_count": self._halt_count,
            "fallback_rate": self._fallback_count / max(self._total_calls, 1),
            "coverage_rate": (
                sum(self._coverage_history) / len(self._coverage_history)
                if self._coverage_history else 0.0
            ),
            "mean_radius": (
                sum(self._radius_history) / len(self._radius_history)
                if self._radius_history else 0.0
            ),
            "current_radius": self.calibrator.current_radius(),
            "calibrator_halted": self.calibrator.is_halted,
            "policy_name": self.policy.name(),
        }

    @property
    def radius_history(self) -> list[float]:
        return self._radius_history

    @property
    def coverage_history(self) -> list[bool]:
        return self._coverage_history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_fallback(self, observation: dict) -> Tensor:
        """Generate a fallback action (identity = no movement)."""
        if self._fallback_action is not None:
            return self._fallback_action.unsqueeze(0)

        image = observation["image"]
        B = image.shape[0]
        device = image.device
        return torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._total_calls = 0
        self._fallback_count = 0
        self._halt_count = 0
        self._radius_history.clear()
        self._coverage_history.clear()

    def __repr__(self) -> str:
        return (
            f"SafePolicyWrapper(policy={self.policy.name()}, "
            f"max_radius={self.max_radius}, "
            f"calls={self._total_calls})"
        )
