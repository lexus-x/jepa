"""Tests for conformal prediction on SE(3)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.conformal.lie_scorer import LieScorer, GeodesicBallPredictor
from src.conformal.online_calibration import OnlineConformalCalibrator
from src.conformal.safe_policy import SafePolicyWrapper
from src.policies.random import RandomPolicy
from src.flow.se3_utils import se3_expmap


def test_lie_scorer_identity():
    """Score of identity vs identity should be zero."""
    scorer = LieScorer()
    T = torch.eye(4).unsqueeze(0)
    score = scorer.score(T, T)
    assert score.item() < 1e-6, f"Expected ~0, got {score.item()}"


def test_lie_scorer_translation():
    """Pure translation should score as translation distance."""
    scorer = LieScorer(weight_rot=0.0, weight_trans=1.0)
    T_pred = torch.eye(4).unsqueeze(0)
    T_true = torch.eye(4).unsqueeze(0)
    T_true[0, 0, 3] = 0.5  # 0.5m translation in x
    score = scorer.score(T_pred, T_true)
    assert abs(score.item() - 0.5) < 1e-4, f"Expected ~0.5, got {score.item()}"


def test_online_calibrator_coverage():
    """After calibration, coverage should be approximately 1-α."""
    calibrator = OnlineConformalCalibrator(alpha=0.1, min_scores=10)

    # Generate calibration data: predictions near ground truth
    T_true = torch.eye(4).unsqueeze(0).expand(50, -1, -1).clone()
    noise = 0.01 * torch.randn(50, 6)
    T_pred = se3_expmap(noise) @ T_true

    # Calibrate
    for i in range(50):
        calibrator.update(T_pred[i:i+1], T_true[i:i+1])

    # Test: should cover ~90% with small noise
    test_true = torch.eye(4).unsqueeze(0).expand(20, -1, -1).clone()
    test_noise = 0.01 * torch.randn(20, 6)
    test_pred = se3_expmap(test_noise) @ test_true

    covered = 0
    for i in range(20):
        result = calibrator.update(test_pred[i:i+1], test_true[i:i+1])
        if result["covered"].item():
            covered += 1

    # With small noise, most should be covered
    assert covered >= 15, f"Expected ≥15 covered, got {covered}/20"


def test_safe_policy_wrapper():
    """SafePolicyWrapper should wrap RandomPolicy without errors."""
    policy = RandomPolicy()
    wrapper = SafePolicyWrapper(policy=policy, max_radius=2.0)

    observation = {"image": torch.randn(1, 3, 64, 64)}
    action, info = wrapper.act(observation, "pick up the cup")

    assert action.shape == (1, 4, 4)
    assert "radius" in info
    assert "fallback" in info
    assert not info["fallback"]  # shouldn't fallback with default settings


def test_safe_policy_fallback_on_halt():
    """Should fallback when calibrator is halted."""
    policy = RandomPolicy()
    calibrator = OnlineConformalCalibrator(alpha=0.1, safety_radius=0.01)
    wrapper = SafePolicyWrapper(policy=policy, calibrator=calibrator, max_radius=0.01)

    # Force halt by making radius exceed safety_radius
    calibrator._halted = True
    calibrator._halt_reason = "test halt"

    observation = {"image": torch.randn(1, 3, 64, 64)}
    action, info = wrapper.act(observation, "test")

    assert info["fallback"]
    assert info["halted"]


def test_geodesic_ball_predictor():
    """GeodesicBallPredictor should calibrate and predict."""
    predictor = GeodesicBallPredictor(alpha=0.1)

    # Calibration data
    T_true = torch.eye(4).unsqueeze(0).expand(30, -1, -1).clone()
    noise = 0.02 * torch.randn(30, 6)
    T_pred = se3_expmap(noise) @ T_true

    radius = predictor.calibrate(T_pred, T_true)
    assert radius > 0, f"Radius should be positive, got {radius}"

    # Prediction set
    test_pred = torch.eye(4).unsqueeze(0)
    samples, mask = predictor.predict(test_pred)
    assert samples.shape[0] == 1


def test_stats_tracking():
    """SafePolicyWrapper should track statistics correctly."""
    policy = RandomPolicy()
    wrapper = SafePolicyWrapper(policy=policy)

    for _ in range(5):
        obs = {"image": torch.randn(1, 3, 64, 64)}
        wrapper.act(obs, "test")

    stats = wrapper.stats
    assert stats["total_calls"] == 5
    assert stats["fallback_rate"] == 0.0
    assert len(wrapper.radius_history) == 5


if __name__ == "__main__":
    test_lie_scorer_identity()
    print("✓ test_lie_scorer_identity")
    test_lie_scorer_translation()
    print("✓ test_lie_scorer_translation")
    test_online_calibrator_coverage()
    print("✓ test_online_calibrator_coverage")
    test_safe_policy_wrapper()
    print("✓ test_safe_policy_wrapper")
    test_safe_policy_fallback_on_halt()
    print("✓ test_safe_policy_fallback_on_halt")
    test_geodesic_ball_predictor()
    print("✓ test_geodesic_ball_predictor")
    test_stats_tracking()
    print("✓ test_stats_tracking")
    print("\nAll tests passed.")
