"""Tests for SE(3) Lie group utilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.flow.se3_utils import (
    se3_expmap, se3_logmap, geodesic_interpolation,
    wrapped_gaussian, se3_geodesic_distance, _se3_inverse,
)


def test_expmap_logmap_roundtrip():
    """exp(log(T)) should recover T."""
    # Generate random SE(3) matrices
    xi = 0.5 * torch.randn(10, 6)
    T = se3_expmap(xi)
    xi_recovered = se3_logmap(T)
    T_recovered = se3_expmap(xi_recovered)

    err = (T - T_recovered).abs().max().item()
    assert err < 1e-4, f"Roundtrip error: {err}"


def test_logmap_expmap_roundtrip():
    """log(exp(ξ)) should recover ξ."""
    xi = 0.3 * torch.randn(10, 6)
    T = se3_expmap(xi)
    xi_recovered = se3_logmap(T)

    err = (xi - xi_recovered).abs().max().item()
    assert err < 1e-4, f"Roundtrip error: {err}"


def test_expmap_identity():
    """exp(0) should be identity."""
    xi = torch.zeros(5, 6)
    T = se3_expmap(xi)
    I = torch.eye(4).unsqueeze(0).expand(5, -1, -1)
    assert (T - I).abs().max().item() < 1e-6


def test_geodesic_interpolation_endpoints():
    """γ(0) = T₀, γ(1) = T₁."""
    T0 = se3_expmap(0.3 * torch.randn(5, 6))
    T1 = se3_expmap(0.3 * torch.randn(5, 6))

    at_0 = geodesic_interpolation(T0, T1, 0.0)
    at_1 = geodesic_interpolation(T0, T1, 1.0)

    err0 = (at_0 - T0).abs().max().item()
    err1 = (at_1 - T1).abs().max().item()

    assert err0 < 1e-5, f"γ(0) != T₀: {err0}"
    assert err1 < 1e-5, f"γ(1) != T₁: {err1}"


def test_geodesic_distance_identity():
    """Distance from T to itself should be zero."""
    T = se3_expmap(0.5 * torch.randn(8, 6))
    d = se3_geodesic_distance(T, T)
    assert d.abs().max().item() < 1e-6


def test_geodesic_distance_symmetry():
    """d(T₀, T₁) = d(T₁, T₀)."""
    T0 = se3_expmap(0.3 * torch.randn(5, 6))
    T1 = se3_expmap(0.3 * torch.randn(5, 6))
    d01 = se3_geodesic_distance(T0, T1)
    d10 = se3_geodesic_distance(T1, T0)
    assert (d01 - d10).abs().max().item() < 1e-5


def test_inverse():
    """T @ T⁻¹ should be identity."""
    T = se3_expmap(0.5 * torch.randn(10, 6))
    T_inv = _se3_inverse(T)
    product = T @ T_inv
    I = torch.eye(4).unsqueeze(0).expand(10, -1, -1)
    assert (product - I).abs().max().item() < 1e-5


def test_wrapped_gaussian_shape():
    """Wrapped Gaussian should produce correct shapes."""
    mean = torch.eye(4).unsqueeze(0).expand(3, -1, -1).clone()
    samples = wrapped_gaussian(mean, sigma_trans=0.1, sigma_rot=0.3, num_samples=10)
    assert samples.shape == (3, 10, 4, 4)


def test_batch_independence():
    """Different batch elements should not interfere."""
    xi = torch.randn(4, 6) * 0.3
    T = se3_expmap(xi)

    # Process individually
    for i in range(4):
        T_single = se3_expmap(xi[i:i+1])
        err = (T[i] - T_single[0]).abs().max().item()
        assert err < 1e-6, f"Batch element {i} differs: {err}"


if __name__ == "__main__":
    tests = [
        test_expmap_logmap_roundtrip,
        test_logmap_expmap_roundtrip,
        test_expmap_identity,
        test_geodesic_interpolation_endpoints,
        test_geodesic_distance_identity,
        test_geodesic_distance_symmetry,
        test_inverse,
        test_wrapped_gaussian_shape,
        test_batch_independence,
    ]
    for t in tests:
        t()
        print(f"✓ {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
