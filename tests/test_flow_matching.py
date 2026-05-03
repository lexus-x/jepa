"""Tests for flow matching on SE(3): loss computation, inference, velocity fields."""

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal flow-matching implementation for testing
# ---------------------------------------------------------------------------


class _DummyVelocityField(nn.Module):
    """Toy velocity field: linear layer on flattened (twist, t) input."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 + 1, hidden),  # 6-dim twist + time
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (..., 6) points in se(3) tangent space.
            t:   (...) time in [0, 1].
        Returns:
            v: (..., 6) velocity vectors.
        """
        t_expanded = t.unsqueeze(-1) if t.dim() < x_t.dim() else t
        inp = torch.cat([x_t, t_expanded], dim=-1)
        return self.net(inp)


def _flow_matching_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Conditional flow matching loss: MSE(v_θ(x_t, t), u_t(x_t | x₁)).

    Uses the optimal transport path: x_t = (1-t) x₀ + t x₁, so u_t = x₁ - x₀.
    """
    x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
    target_velocity = x1 - x0
    pred_velocity = model(x_t, t)
    return torch.mean((pred_velocity - target_velocity) ** 2)


def _euler_integrate(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """Simple Euler integration from x0 at t=0 to t=1."""
    dt = 1.0 / steps
    x = x0.clone()
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v = model(x, t)
        x = x + v * dt
    return x


def _geodesic_path_samples(
    x0: torch.Tensor,
    x1: torch.Tensor,
    num_points: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points along the geodesic (linear in tangent space) path.

    Returns:
        x_t: (num_points, batch, 6) points along path.
        t:   (num_points,) corresponding times.
    """
    ts = torch.linspace(0, 1, num_points)
    points = []
    for ti in ts:
        x_t = (1 - ti) * x0 + ti * x1
        points.append(x_t)
    return torch.stack(points), ts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlowMatchingLoss:
    """Flow matching loss should decrease with training."""

    def test_loss_shape(self):
        """Loss should be a scalar."""
        model = _DummyVelocityField()
        x0 = torch.randn(8, 6)
        x1 = torch.randn(8, 6)
        t = torch.rand(8)
        loss = _flow_matching_loss(model, x0, x1, t)
        assert loss.shape == ()

    def test_loss_finite(self):
        """Loss should be finite (not NaN or Inf)."""
        model = _DummyVelocityField()
        x0 = torch.randn(16, 6)
        x1 = torch.randn(16, 6)
        t = torch.rand(16)
        loss = _flow_matching_loss(model, x0, x1, t)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_training(self):
        """After a few gradient steps, loss should decrease."""
        torch.manual_seed(0)
        model = _DummyVelocityField(hidden=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x0 = torch.randn(32, 6)
        x1 = x0 + torch.randn(32, 6) * 0.5  # nearby targets

        losses = []
        for _ in range(50):
            t = torch.rand(32)
            loss = _flow_matching_loss(model, x0, x1, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should trend downward (compare first 10 vs last 10)
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"


class TestFlowInference:
    """Trained flow should map x0 close to x1."""

    def test_inference_shape(self):
        """Euler integration output shape matches input."""
        model = _DummyVelocityField()
        x0 = torch.randn(4, 6)
        x_out = _euler_integrate(model, x0, steps=10)
        assert x_out.shape == x0.shape

    def test_inference_finite(self):
        """Integration output should be finite."""
        model = _DummyVelocityField()
        x0 = torch.randn(8, 6)
        x_out = _euler_integrate(model, x0, steps=20)
        assert torch.isfinite(x_out).all()

    def test_trained_flow_maps_correctly(self):
        """After training, flow should map x0 → x1 reasonably well."""
        torch.manual_seed(42)
        model = _DummyVelocityField(hidden=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # Generate fixed data
        x0 = torch.randn(64, 6)
        x1 = x0 + torch.randn(64, 6) * 0.3

        # Train
        for _ in range(300):
            t = torch.rand(64)
            loss = _flow_matching_loss(model, x0, x1, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on training data
        with torch.no_grad():
            x_pred = _euler_integrate(model, x0, steps=100)
        mse = ((x_pred - x1) ** 2).mean().item()
        assert mse < 0.5, f"MSE too high after training: {mse:.4f}"


class TestVelocityFieldShapes:
    """Velocity field output shapes should match input shapes."""

    def test_single_sample(self):
        model = _DummyVelocityField()
        x = torch.randn(1, 6)
        t = torch.tensor([0.5])
        v = model(x, t)
        assert v.shape == (1, 6)

    def test_batch(self):
        model = _DummyVelocityField()
        x = torch.randn(16, 6)
        t = torch.rand(16)
        v = model(x, t)
        assert v.shape == (16, 6)

    def test_different_batch_sizes(self):
        model = _DummyVelocityField()
        for bs in [1, 4, 32]:
            x = torch.randn(bs, 6)
            t = torch.rand(bs)
            v = model(x, t)
            assert v.shape == (bs, 6)


class TestGeodesicPathSampling:
    """Geodesic path samples should interpolate between endpoints."""

    def test_path_endpoints(self):
        """First sample ≈ x0, last sample ≈ x1."""
        x0 = torch.randn(4, 6)
        x1 = torch.randn(4, 6)
        path, ts = _geodesic_path_samples(x0, x1, num_points=11)
        torch.testing.assert_close(path[0], x0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(path[-1], x1, atol=1e-6, rtol=1e-6)

    def test_path_monotonic(self):
        """Samples should be ordered by time."""
        _, ts = _geodesic_path_samples(torch.zeros(2, 6), torch.ones(2, 6), num_points=20)
        diffs = ts[1:] - ts[:-1]
        assert (diffs >= 0).all()

    def test_path_shape(self):
        """Path shape: (num_points, batch, 6)."""
        path, ts = _geodesic_path_samples(torch.randn(8, 6), torch.randn(8, 6), num_points=15)
        assert path.shape == (15, 8, 6)
        assert ts.shape == (15,)

    def test_path_interpolation_linear(self):
        """Midpoint should be average of endpoints (linear interpolation)."""
        x0 = torch.zeros(2, 6)
        x1 = torch.ones(2, 6) * 4
        path, _ = _geodesic_path_samples(x0, x1, num_points=3)
        torch.testing.assert_close(path[1], torch.ones(2, 6) * 2, atol=1e-6, rtol=1e-6)
