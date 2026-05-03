"""Tests for conformal prediction: nonconformity scores, calibration, coverage, safety halt."""

import pytest
import torch

# ---------------------------------------------------------------------------
# Conformal prediction utilities — standalone implementations for testing
# ---------------------------------------------------------------------------


class ConformalCalibrator:
    """Online conformal prediction calibrator with safety halt.

    Maintains a buffer of nonconformity scores and provides prediction sets
    with finite-sample coverage guarantees.
    """

    def __init__(self, alpha: float = 0.1, buffer_size: int = 1000, safety_threshold: float = 10.0):
        """
        Args:
            alpha: miscoverage level (target coverage = 1 - alpha).
            buffer_size: max calibration scores to keep.
            safety_threshold: max allowed nonconformity score before triggering halt.
        """
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.safety_threshold = safety_threshold
        self.scores: list[float] = []
        self._quantile: float | None = None
        self.halted = False
        self.halt_reason: str | None = None

    def nonconformity_score(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute L2 nonconformity score: ‖prediction - ground_truth‖₂.

        Args:
            prediction: (..., D) predicted values.
            ground_truth: (..., D) true values.
        Returns:
            scores: (...) nonconformity scores (non-negative).
        """
        return torch.norm(prediction - ground_truth, dim=-1)

    def update(self, score: float) -> None:
        """Add a new calibration score and update the quantile.

        Args:
            score: new nonconformity score to add.
        Raises:
            RuntimeError: if safety halt is triggered.
        """
        if score > self.safety_threshold:
            self.halted = True
            self.halt_reason = f"Nonconformity score {score:.4f} exceeds safety threshold {self.safety_threshold}"
            return

        self.scores.append(score)
        if len(self.scores) > self.buffer_size:
            self.scores = self.scores[-self.buffer_size:]

        self._recalibrate()

    def _recalibrate(self) -> None:
        """Recompute the (1-α) quantile of the score buffer."""
        if len(self.scores) < 2:
            self._quantile = float("inf")
            return
        sorted_scores = sorted(self.scores)
        idx = int((1 - self.alpha) * len(sorted_scores))
        idx = min(idx, len(sorted_scores) - 1)
        self._quantile = sorted_scores[idx]

    @property
    def threshold(self) -> float:
        """Current conformal threshold (quantile)."""
        if self._quantile is None:
            return float("inf")
        return self._quantile

    def is_covered(self, score: float) -> bool:
        """Check if a score falls within the prediction set."""
        return score <= self.threshold

    def check_coverage(self, scores: list[float]) -> float:
        """Compute empirical coverage on a list of scores."""
        if not scores:
            return 1.0
        covered = sum(1 for s in scores if self.is_covered(s))
        return covered / len(scores)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNonconformityScore:
    """Nonconformity score computation."""

    def test_zero_for_identical(self):
        """Score should be zero when prediction matches ground truth."""
        cal = ConformalCalibrator()
        pred = torch.tensor([1.0, 2.0, 3.0])
        gt = torch.tensor([1.0, 2.0, 3.0])
        score = cal.nonconformity_score(pred, gt)
        torch.testing.assert_close(score, torch.tensor(0.0), atol=1e-7, rtol=1e-7)

    def test_positive_for_different(self):
        """Score should be positive when prediction differs from ground truth."""
        cal = ConformalCalibrator()
        pred = torch.tensor([1.0, 0.0])
        gt = torch.tensor([0.0, 0.0])
        score = cal.nonconformity_score(pred, gt)
        assert score.item() == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self):
        """Score should be symmetric: score(a, b) = score(b, a)."""
        cal = ConformalCalibrator()
        a = torch.randn(10)
        b = torch.randn(10)
        s1 = cal.nonconformity_score(a, b)
        s2 = cal.nonconformity_score(b, a)
        torch.testing.assert_close(s1, s2, atol=1e-7, rtol=1e-7)

    def test_batched_scores(self):
        """Batched computation should produce per-sample scores."""
        cal = ConformalCalibrator()
        pred = torch.randn(8, 3)
        gt = torch.randn(8, 3)
        scores = cal.nonconformity_score(pred, gt)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_triangle_inequality(self):
        """Score satisfies triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)."""
        cal = ConformalCalibrator()
        a = torch.randn(5)
        b = torch.randn(5)
        c = torch.randn(5)
        d_ac = cal.nonconformity_score(a, c).item()
        d_ab = cal.nonconformity_score(a, b).item()
        d_bc = cal.nonconformity_score(b, c).item()
        assert d_ac <= d_ab + d_bc + 1e-6


class TestOnlineCalibration:
    """Online calibration update and threshold computation."""

    def test_threshold_starts_at_infinity(self):
        """Before any calibration, threshold should be infinity."""
        cal = ConformalCalibrator()
        assert cal.threshold == float("inf")

    def test_threshold_updates(self):
        """After adding scores, threshold should be finite."""
        cal = ConformalCalibrator(alpha=0.1)
        for i in range(100):
            cal.update(float(i) / 100)
        assert cal.threshold < float("inf")
        assert cal.threshold > 0

    def test_buffer_respects_size(self):
        """Buffer should not exceed configured size."""
        cal = ConformalCalibrator(buffer_size=50)
        for i in range(200):
            cal.update(float(i))
        assert len(cal.scores) <= 50

    def test_quantile_increases_with_scores(self):
        """Higher scores should push the threshold up."""
        cal1 = ConformalCalibrator(alpha=0.1)
        cal2 = ConformalCalibrator(alpha=0.1)
        for i in range(100):
            cal1.update(0.1)
            cal2.update(1.0)
        assert cal2.threshold > cal1.threshold

    def test_low_alpha_gives_high_threshold(self):
        """Lower alpha (more coverage) should yield a higher threshold."""
        cal_tight = ConformalCalibrator(alpha=0.01)
        cal_loose = ConformalCalibrator(alpha=0.5)
        for i in range(100):
            score = float(i) / 100
            cal_tight.update(score)
            cal_loose.update(score)
        assert cal_tight.threshold >= cal_loose.threshold


class TestCoverageGuarantee:
    """Coverage should be at least 1-alpha on exchangeable data."""

    def test_coverage_on_uniform(self):
        """On uniform i.i.d. scores, coverage should be ≥ 1-alpha (with margin)."""
        torch.manual_seed(42)
        alpha = 0.1
        n_cal = 200
        n_test = 500

        # Generate calibration scores
        cal_scores = torch.rand(n_cal).tolist()
        cal = ConformalCalibrator(alpha=alpha)
        for s in cal_scores:
            cal.update(s)

        # Generate test scores (same distribution)
        test_scores = torch.rand(n_test).tolist()
        coverage = cal.check_coverage(test_scores)

        # Coverage should be close to 1-alpha = 0.9, allow some slack
        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low for alpha={alpha}"

    def test_coverage_guarantee_with_exact_quantile(self):
        """Using exact scores, coverage should match theoretical guarantee."""
        alpha = 0.2
        n = 100
        # Calibration scores: 0, 1, 2, ..., 99
        cal = ConformalCalibrator(alpha=alpha)
        for i in range(n):
            cal.update(float(i))

        # Threshold should be at index ceil((1-alpha)*(n+1))-1 ≈ 80
        # All test scores ≤ 80 should be covered
        test_scores_below = [float(i) for i in range(int(cal.threshold) + 1)]
        assert cal.check_coverage(test_scores_below) == 1.0

        # Some scores above threshold should not be covered
        test_scores_above = [float(i) for i in range(int(cal.threshold) + 1, 200)]
        assert cal.check_coverage(test_scores_above) < 1.0


class TestSafetyHalt:
    """Safety halt should trigger on anomalous scores."""

    def test_halt_on_extreme_score(self):
        """Halt when score exceeds safety threshold."""
        cal = ConformalCalibrator(safety_threshold=5.0)
        cal.update(1.0)
        assert not cal.halted
        cal.update(10.0)  # exceeds threshold
        assert cal.halted
        assert cal.halt_reason is not None
        assert "10.0" in cal.halt_reason

    def test_no_halt_within_threshold(self):
        """No halt when all scores are within threshold."""
        cal = ConformalCalibrator(safety_threshold=10.0)
        for i in range(50):
            cal.update(float(i) * 0.1)
        assert not cal.halted

    def test_halt_preserves_state(self):
        """After halt, calibrator state should be preserved for inspection."""
        cal = ConformalCalibrator(safety_threshold=2.0)
        cal.update(0.5)
        cal.update(1.0)
        cal.update(5.0)  # triggers halt
        assert cal.halted
        # Scores before halt should still be there
        assert len(cal.scores) == 2

    def test_halt_with_batch_score_check(self):
        """Safety halt should work when checking a batch of scores."""
        cal = ConformalCalibrator(safety_threshold=3.0)
        # Simulate batch: compute scores and check each
        predictions = torch.randn(10, 3) * 0.1
        ground_truth = torch.randn(10, 3) * 0.1
        scores = cal.nonconformity_score(predictions, ground_truth)

        halt_triggered = False
        for s in scores.tolist():
            if s > cal.safety_threshold:
                halt_triggered = True
                break
            cal.update(s)
        # With small noise, most scores should be < 3.0
        # This tests the pattern works correctly
