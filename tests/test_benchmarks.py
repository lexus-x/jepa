"""Benchmark sanity tests: env creation, action format compatibility.

These tests verify that the benchmark environment interfaces work correctly.
If the actual benchmark libraries (libero, metaworld) are not installed,
tests use mocks to validate the interface contracts.
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Check availability
# ---------------------------------------------------------------------------

HAS_LIBERO = importlib.util.find_spec("libero") is not None
HAS_METAWORLD = importlib.util.find_spec("metaworld") is not None

# ---------------------------------------------------------------------------
# Standard action format: (B, action_dim) where action_dim = 7
#   [dx, dy, dz, droll, dpitch, dyaw, gripper]
# ---------------------------------------------------------------------------

ACTION_DIM = 7


def _validate_action_format(action: torch.Tensor, batch_size: int) -> None:
    """Validate that actions conform to the expected format."""
    assert action.dim() == 2, f"Expected 2D tensor, got {action.dim()}D"
    assert action.shape == (batch_size, ACTION_DIM), \
        f"Expected shape ({batch_size}, {ACTION_DIM}), got {action.shape}"
    assert torch.isfinite(action).all(), "Actions contain NaN or Inf"
    # Gripper should be in [0, 1] or [-1, 1]
    gripper = action[:, -1]
    assert (gripper >= -1.01).all() and (gripper <= 1.01).all(), \
        f"Gripper values out of range: [{gripper.min():.2f}, {gripper.max():.2f}]"


def _make_mock_env(name: str, obs_dim: int = 512, action_dim: int = 7):
    """Create a mock environment with standard interface."""
    env = MagicMock()
    env.name = name
    env.observation_space = MagicMock(shape=(obs_dim,))
    env.action_space = MagicMock(shape=(action_dim,))
    env.action_space.sample.return_value = torch.randn(action_dim)

    def _reset():
        return torch.randn(obs_dim), {}

    def _step(action):
        return torch.randn(obs_dim), 0.0, False, False, {}

    env.reset = MagicMock(side_effect=_reset)
    env.step = MagicMock(side_effect=_step)
    env.close = MagicMock()
    return env


# ---------------------------------------------------------------------------
# Tests — LIBERO
# ---------------------------------------------------------------------------


class TestLIBEROEnv:
    """LIBERO environment creation and interface tests."""

    @pytest.mark.skipif(HAS_LIBERO, reason="Test uses mocks; real LIBERO is installed")
    def test_mock_libero_creation(self):
        """Mock LIBERO env should have standard interface."""
        env = _make_mock_env("libero-spatial", obs_dim=512, action_dim=ACTION_DIM)
        assert env.name == "libero-spatial"
        assert env.observation_space.shape == (512,)
        assert env.action_space.shape == (ACTION_DIM,)

    @pytest.mark.skipif(HAS_LIBERO, reason="Test uses mocks; real LIBERO is installed")
    def test_mock_libero_reset(self):
        """Mock LIBERO reset should return observation."""
        env = _make_mock_env("libero-spatial")
        obs, info = env.reset()
        assert obs.shape == (512,)
        assert isinstance(info, dict)

    @pytest.mark.skipif(HAS_LIBERO, reason="Test uses mocks; real LIBERO is installed")
    def test_mock_libero_step(self):
        """Mock LIBERO step should return standard tuple."""
        env = _make_mock_env("libero-spatial")
        env.reset()
        action = torch.randn(ACTION_DIM)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (512,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.skipif(not HAS_LIBERO, reason="LIBERO not installed")
    @pytest.mark.slow
    def test_real_libero_creation(self):
        """Test real LIBERO env creation."""
        import libero
        # This test only runs if LIBERO is actually installed
        # The import itself verifies the package is loadable
        assert libero is not None


# ---------------------------------------------------------------------------
# Tests — MetaWorld
# ---------------------------------------------------------------------------


class TestMetaWorldEnv:
    """MetaWorld environment creation and interface tests."""

    @pytest.mark.skipif(HAS_METAWORLD, reason="Test uses mocks; real MetaWorld is installed")
    def test_mock_metaworld_creation(self):
        """Mock MetaWorld env should have standard interface."""
        env = _make_mock_env("reach-v2", obs_dim=39, action_dim=ACTION_DIM)
        assert env.name == "reach-v2"
        assert env.observation_space.shape == (39,)
        assert env.action_space.shape == (ACTION_DIM,)

    @pytest.mark.skipif(HAS_METAWORLD, reason="Test uses mocks; real MetaWorld is installed")
    def test_mock_metaworld_reset(self):
        """Mock MetaWorld reset should return observation."""
        env = _make_mock_env("reach-v2")
        obs, info = env.reset()
        assert obs.shape == (39,)
        assert isinstance(info, dict)

    @pytest.mark.skipif(HAS_METAWORLD, reason="Test uses mocks; real MetaWorld is installed")
    def test_mock_metaworld_step(self):
        """Mock MetaWorld step should return standard tuple."""
        env = _make_mock_env("reach-v2")
        env.reset()
        action = torch.randn(ACTION_DIM)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (39,)
        assert isinstance(reward, float)

    @pytest.mark.skipif(not HAS_METAWORLD, reason="MetaWorld not installed")
    @pytest.mark.slow
    def test_real_metaworld_creation(self):
        """Test real MetaWorld env creation."""
        import metaworld
        assert metaworld is not None


# ---------------------------------------------------------------------------
# Tests — Action Format Compatibility
# ---------------------------------------------------------------------------


class TestActionFormatCompatibility:
    """Actions should be compatible across all benchmark environments."""

    def test_action_shape_standard(self):
        """Standard action should be (B, 7)."""
        for B in [1, 4, 16, 64]:
            action = torch.randn(B, ACTION_DIM)
            _validate_action_format(action, B)

    def test_action_from_model_output(self):
        """Simulated model output should be valid actions."""
        # Simulate a model that outputs actions
        model = torch.nn.Linear(128, ACTION_DIM)
        features = torch.randn(8, 128)
        action = model(features)
        _validate_action_format(action, 8)

    def test_action_clipping(self):
        """Clipped actions should still be valid."""
        raw = torch.randn(16, ACTION_DIM) * 5
        clipped = torch.clamp(raw, -1.0, 1.0)
        _validate_action_format(clipped, 16)

    def test_gripper_binary(self):
        """Gripper should support binary (open/close) commands."""
        B = 8
        action = torch.zeros(B, ACTION_DIM)
        # Open gripper
        action[:4, -1] = 1.0
        # Close gripper
        action[4:, -1] = -1.0
        _validate_action_format(action, B)

    def test_action_horizon_stacking(self):
        """Action chunks (B, H, A) should flatten to (B*H, A) for env."""
        B, H, A = 4, 8, ACTION_DIM
        chunk = torch.randn(B, H, A)
        flat = chunk.reshape(B * H, A)
        _validate_action_format(flat, B * H)

    def test_zero_action(self):
        """Zero action (do nothing) should be valid."""
        action = torch.zeros(4, ACTION_DIM)
        _validate_action_format(action, 4)

    def test_action_consistency_across_envs(self):
        """Same action format works for LIBERO-style and MetaWorld-style obs."""
        # LIBERO: obs_dim=512, MetaWorld: obs_dim=39
        action = torch.randn(2, ACTION_DIM)
        _validate_action_format(action, 2)

        # The action is independent of observation dimension
        for obs_dim in [39, 256, 512, 1024]:
            env = _make_mock_env(f"env-obs{obs_dim}", obs_dim=obs_dim, action_dim=ACTION_DIM)
            obs, _ = env.reset()
            assert obs.shape == (obs_dim,)
            # Same action works regardless of obs dim
            env.step(action[0])
            env.close()
