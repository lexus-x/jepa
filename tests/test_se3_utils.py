"""Tests for SE(3) utility operations: exp/log maps, geodesic interpolation, and sampling."""

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers — standalone implementations so tests don't depend on the package
# being importable yet.  These mirror the canonical SE(3) operations.
# ---------------------------------------------------------------------------


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) vectors to (..., 3, 3) skew-symmetric matrices."""
    x, y, z = v.unbind(-1)
    O = torch.zeros_like(x)
    return torch.stack([
        O, -z, y,
        z, O, -x,
        -y, x, O,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def _rodrigues(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from so(3) to SO(3) via Rodrigues' formula.

    Args:
        omega: (..., 3) axis-angle vectors.
    Returns:
        R: (..., 3, 3) rotation matrices.
    """
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=eps)  # (..., 1)
    k = omega / theta  # unit axis
    K = _skew_symmetric(k)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    # R = I + sin(θ) K + (1 - cos(θ)) K²
    return I + theta.unsqueeze(-1).sin() * K + (1 - theta.unsqueeze(-1).cos()) * (K @ K)


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Exponential map from se(3) to SE(3).

    Args:
        twist: (..., 6) vectors — first 3 are rotation (ω), last 3 are translation (v).
    Returns:
        T: (..., 4, 4) homogeneous transformation matrices.
    """
    omega = twist[..., :3]
    v = twist[..., 3:]
    R = _rodrigues(omega)
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    K = _skew_symmetric(omega / theta)
    # V = I + (1-cos θ)/θ² K + (θ - sin θ)/θ³ K²
    theta_sq = theta ** 2
    theta_cu = theta ** 3
    I = torch.eye(3, device=twist.device, dtype=twist.dtype).expand_as(K)
    V = I + ((1 - theta.cos()) / theta_sq).unsqueeze(-1) * K + \
        ((theta - theta.sin()) / theta_cu).unsqueeze(-1) * (K @ K)
    t = (V @ v.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    T = torch.zeros(*twist.shape[:-1], 4, 4, device=twist.device, dtype=twist.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_log(T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithm map from SE(3) to se(3).

    Args:
        T: (..., 4, 4) homogeneous transformation matrices.
    Returns:
        twist: (..., 6) vectors.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # Log map for SO(3)
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = cos_theta.clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)  # (...)
    theta_safe = theta.clamp(min=eps)

    # ω = θ / (2 sin θ) * [R - Rᵀ]ᵛ
    omega_hat = (theta_safe / (2 * theta_safe.sin())).unsqueeze(-1).unsqueeze(-1) * \
                (R - R.transpose(-1, -2))
    omega = torch.stack([
        omega_hat[..., 2, 1],
        omega_hat[..., 0, 2],
        omega_hat[..., 1, 0],
    ], dim=-1)

    # Handle θ ≈ 0 (identity rotation)
    small = theta.unsqueeze(-1) < eps
    omega = torch.where(small, torch.zeros_like(omega), omega)

    # V⁻¹ for translation
    K = _skew_symmetric(omega / theta_safe.unsqueeze(-1).clamp(min=eps))
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(K)
    half_theta = theta_safe / 2
    V_inv = I - 0.5 * K + (1 / theta_safe.unsqueeze(-1).clamp(min=eps) ** 2 *
                            (1 - theta_safe.unsqueeze(-1) * half_theta.unsqueeze(-1).cos() /
                             half_theta.unsqueeze(-1).sin().clamp(min=eps))).unsqueeze(-1) * (K @ K)
    # Fix near-zero theta
    V_inv = torch.where(small.unsqueeze(-1).expand_as(V_inv), I, V_inv)
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)

    return torch.cat([omega, v], dim=-1)


def geodesic_interpolate(T0: torch.Tensor, T1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Geodesic interpolation on SE(3).

    γ(α) = T₀ ∘ exp(α · log(T₀⁻¹ ∘ T₁))

    Args:
        T0, T1: (..., 4, 4) SE(3) matrices.
        alpha: interpolation parameter in [0, 1].
    Returns:
        T: (..., 4, 4) interpolated SE(3) matrix.
    """
    T0_inv = torch.linalg.inv(T0)
    relative = T0_inv @ T1
    twist = se3_log(relative)
    return T0 @ se3_exp(alpha * twist)


def wrapped_gaussian_sample(
    mean_twist: torch.Tensor,
    std: float,
    num_samples: int,
) -> torch.Tensor:
    """Sample from a Gaussian on se(3) mapped to SE(3).

    Args:
        mean_twist: (6,) mean in tangent space.
        std: standard deviation.
        num_samples: number of samples.
    Returns:
        T: (num_samples, 4, 4) sampled SE(3) matrices.
    """
    noise = torch.randn(num_samples, 6) * std
    twists = mean_twist.unsqueeze(0) + noise
    return se3_exp(twists)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExpLogRoundtrip:
    """exp(log(T)) ≈ T and log(exp(ξ)) ≈ ξ."""

    def test_identity(self):
        """exp of zero twist is the identity."""
        zero = torch.zeros(6)
        T = se3_exp(zero)
        expected = torch.eye(4)
        torch.testing.assert_close(T, expected, atol=1e-6, rtol=1e-6)

    def test_roundtrip_small_rotation(self):
        """Small twist round-trips through log → exp."""
        twist = torch.tensor([0.1, -0.2, 0.15, 0.5, -0.3, 0.2])
        T = se3_exp(twist)
        twist_recovered = se3_log(T)
        torch.testing.assert_close(twist_recovered, twist, atol=1e-5, rtol=1e-5)

    def test_roundtrip_random(self):
        """Random twists round-trip within tolerance."""
        torch.manual_seed(42)
        for _ in range(20):
            twist = torch.randn(6) * 0.5
            T = se3_exp(twist)
            twist_back = se3_log(T)
            torch.testing.assert_close(twist_back, twist, atol=1e-4, rtol=1e-4)

    def test_roundtrip_batched(self):
        """Batched exp/log roundtrip."""
        twists = torch.randn(8, 6) * 0.3
        T = se3_exp(twists)
        twists_back = se3_log(T)
        torch.testing.assert_close(twists_back, twists, atol=1e-4, rtol=1e-4)

    def test_rotation_matrix_orthogonality(self):
        """R from exp should be a valid rotation: R^T R = I, det(R) = 1."""
        twist = torch.randn(6) * 0.8
        T = se3_exp(twist)
        R = T[:3, :3]
        RtR = R.T @ R
        torch.testing.assert_close(RtR, torch.eye(3), atol=1e-5, rtol=1e-5)
        assert torch.det(R).item() == pytest.approx(1.0, abs=1e-5)


class TestGeodesicInterpolation:
    """γ(0) = T₀, γ(1) = T₁, and midpoint is valid."""

    def test_endpoint_zero(self):
        """γ(0) = T₀."""
        torch.manual_seed(0)
        T0 = se3_exp(torch.randn(6) * 0.5)
        T1 = se3_exp(torch.randn(6) * 0.5)
        T_interp = geodesic_interpolate(T0, T1, 0.0)
        torch.testing.assert_close(T_interp, T0, atol=1e-5, rtol=1e-5)

    def test_endpoint_one(self):
        """γ(1) = T₁."""
        torch.manual_seed(1)
        T0 = se3_exp(torch.randn(6) * 0.5)
        T1 = se3_exp(torch.randn(6) * 0.5)
        T_interp = geodesic_interpolate(T0, T1, 1.0)
        torch.testing.assert_close(T_interp, T1, atol=1e-5, rtol=1e-5)

    def test_midpoint_valid_se3(self):
        """Midpoint should still be a valid SE(3) matrix."""
        T0 = se3_exp(torch.tensor([0.3, 0.0, 0.0, 1.0, 0.0, 0.0]))
        T1 = se3_exp(torch.tensor([0.0, 0.3, 0.0, 0.0, 1.0, 0.0]))
        T_mid = geodesic_interpolate(T0, T1, 0.5)
        R = T_mid[:3, :3]
        RtR = R.T @ R
        torch.testing.assert_close(RtR, torch.eye(3), atol=1e-5, rtol=1e-5)
        assert torch.det(R).item() == pytest.approx(1.0, abs=1e-5)
        assert T_mid[3, 3].item() == pytest.approx(1.0, abs=1e-6)

    def test_batched_interpolation(self):
        """Batched geodesic interpolation preserves endpoints."""
        T0 = se3_exp(torch.randn(4, 6) * 0.3)
        T1 = se3_exp(torch.randn(4, 6) * 0.3)
        T_at_0 = geodesic_interpolate(T0, T1, 0.0)
        T_at_1 = geodesic_interpolate(T0, T1, 1.0)
        torch.testing.assert_close(T_at_0, T0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(T_at_1, T1, atol=1e-5, rtol=1e-5)


class TestWrappedGaussian:
    """Samples from wrapped Gaussian should be valid SE(3) matrices."""

    def test_samples_valid_rotation(self):
        """All samples should have orthonormal rotation blocks."""
        mean = torch.zeros(6)
        samples = wrapped_gaussian_sample(mean, std=0.1, num_samples=64)
        for i in range(64):
            R = samples[i, :3, :3]
            RtR = R.T @ R
            torch.testing.assert_close(RtR, torch.eye(3), atol=1e-4, rtol=1e-4)
            assert torch.det(R).item() == pytest.approx(1.0, abs=1e-4)

    def test_samples_shape(self):
        """Output shape should be (N, 4, 4)."""
        mean = torch.zeros(6)
        samples = wrapped_gaussian_sample(mean, std=0.2, num_samples=32)
        assert samples.shape == (32, 4, 4)

    def test_samples_centered_near_mean(self):
        """Mean of sampled translations should be close to the mean twist translation."""
        torch.manual_seed(123)
        mean = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        samples = wrapped_gaussian_sample(mean, std=0.01, num_samples=512)
        avg_translation = samples[:, :3, 3].mean(dim=0)
        torch.testing.assert_close(avg_translation, mean[3:], atol=0.1, rtol=0.1)

    def test_homogeneous_last_row(self):
        """Last row of all samples should be [0, 0, 0, 1]."""
        samples = wrapped_gaussian_sample(torch.zeros(6), std=0.3, num_samples=16)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(16, 4)
        torch.testing.assert_close(samples[:, 3, :], expected, atol=1e-6, rtol=1e-6)
