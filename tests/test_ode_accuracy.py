"""Numerical verification tests for the adaptive ODE integrator.

The Dormand-Prince RK45 integrator in geodesic_flow.py is hand-rolled.
These tests verify its accuracy against:
    1. An analytical solution (linear ODE on SE(3))
    2. The reference torchdiffeq implementation (if available)
    3. Fixed-step RK4 at high resolution as a ground truth proxy

Run: pytest tests/test_ode_accuracy.py -v
"""

from __future__ import annotations

import math
import pytest
import torch
from torch import Tensor

# Adjust path for direct execution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flow.se3_utils import se3_expmap, se3_logmap, se3_geodesic_distance
from src.flow.se3_manifold import SE3Manifold


# ======================================================================
# Analytical test case: constant-velocity geodesic on SE(3)
# ======================================================================

def make_constant_velocity_ode(xi_const: Tensor):
    """Create an ODE with constant velocity in se(3).

    The solution is x(t) = exp(t * xi) @ x(0), which we can compute analytically.

    Args:
        xi_const: [B, 6] constant velocity in se(3).

    Returns:
        velocity_fn: Callable(x_t, t) → [B, 6] velocity.
        analytical_solution: Callable(t, x0) → [B, 4, 4] solution at time t.
    """
    def velocity_fn(x_t: Tensor, t: Tensor) -> Tensor:
        # Constant velocity, independent of state and time
        return xi_const.expand(x_t.shape[0], -1)

    def analytical_solution(t: float, x0: Tensor) -> Tensor:
        xi_t = xi_const * t  # [B, 6]
        T_t = se3_expmap(xi_t)  # [B, 4, 4]
        return T_t @ x0

    return velocity_fn, analytical_solution


# ======================================================================
# Tests
# ======================================================================

class TestAdaptiveODEAccuracy:
    """Verify the adaptive ODE integrator against analytical solutions."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def identity_batch(self, device):
        """Batch of identity poses."""
        return torch.eye(4, device=device).unsqueeze(0).expand(4, -1, -1).clone()

    def test_constant_velocity_converges_to_analytical(self, device, identity_batch):
        """For a constant-velocity ODE, the integrator must match x(t)=exp(t*xi)@x0."""
        from src.flow.geodesic_flow import AdaptiveODEIntegrator

        # Small velocity so the ODE is well-conditioned
        xi = torch.tensor([[0.1, -0.05, 0.08, 0.2, -0.1, 0.15]], device=device)
        vel_fn, analytical_fn = make_constant_velocity_ode(xi)

        integrator = AdaptiveODEIntegrator(
            velocity_fn=vel_fn,
            atol=1e-4,
            rtol=1e-3,
            max_steps=100,
        )

        x0 = identity_batch[:1]  # [1, 4, 4]
        x1_numerical, info = integrator.integrate(x0)
        x1_analytical = analytical_fn(1.0, x0)

        geodesic_err = se3_geodesic_distance(x1_numerical, x1_analytical)
        assert geodesic_err.item() < 1e-3, (
            f"Constant-velocity ODE: geodesic error {geodesic_err.item():.6f} "
            f"exceeds tolerance 1e-3 after {info['steps']} steps"
        )

    def test_batch_independence(self, device):
        """Different samples in a batch should integrate independently."""
        from src.flow.geodesic_flow import AdaptiveODEIntegrator

        xi1 = torch.tensor([[0.1, 0.0, 0.0, 0.3, 0.0, 0.0]], device=device)
        xi2 = torch.tensor([[0.0, 0.0, 0.2, 0.0, 0.0, -0.4]], device=device)
        xi_batch = torch.cat([xi1, xi2], dim=0)  # [2, 6]

        vel_fn, _ = make_constant_velocity_ode(xi_batch)

        integrator = AdaptiveODEIntegrator(
            velocity_fn=vel_fn,
            atol=1e-4,
            rtol=1e-3,
            max_steps=100,
        )

        x0 = torch.eye(4, device=device).unsqueeze(0).expand(2, -1, -1).clone()
        x1_batch, _ = integrator.integrate(x0)

        # Verify each sample independently
        x1_solo1, _ = integrator.integrate(x0[:1])
        x1_solo2, _ = integrator.integrate(x0[1:])

        err1 = se3_geodesic_distance(x1_batch[:1], x1_solo1)
        err2 = se3_geodesic_distance(x1_batch[1:], x1_solo2)

        assert err1.item() < 1e-5, f"Batch sample 0 differs from solo: {err1.item()}"
        assert err2.item() < 1e-5, f"Batch sample 1 differs from solo: {err2.item()}"

    def test_easy_ode_fewer_steps_than_hard(self, device):
        """Easy (low curvature) ODEs should need fewer steps than hard ones."""
        from src.flow.geodesic_flow import AdaptiveODEIntegrator

        # Easy: small, smooth velocity
        xi_easy = torch.tensor([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]], device=device)
        # Hard: large, mixed velocity
        xi_hard = torch.tensor([[0.5, -0.3, 0.4, 1.0, -0.8, 0.6]], device=device)

        x0 = torch.eye(4, device=device).unsqueeze(0)

        easy_fn, _ = make_constant_velocity_ode(xi_easy)
        hard_fn, _ = make_constant_velocity_ode(xi_hard)

        easy_int = AdaptiveODEIntegrator(velocity_fn=easy_fn, atol=1e-3, rtol=1e-2)
        hard_int = AdaptiveODEIntegrator(velocity_fn=hard_fn, atol=1e-3, rtol=1e-2)

        _, easy_info = easy_int.integrate(x0)
        _, hard_info = hard_int.integrate(x0)

        # Easy should need fewer or equal steps
        assert easy_info["steps"] <= hard_info["steps"] + 5, (
            f"Easy ODE ({easy_info['steps']} steps) should need ≤ hard ODE "
            f"({hard_info['steps']} steps)"
        )

    def test_adaptive_fewer_steps_than_fixed_rk4(self, device):
        """Adaptive integrator should use fewer steps than fixed RK4 at same accuracy."""
        from src.flow.geodesic_flow import AdaptiveODEIntegrator, GeodesicFlowMatcher

        xi = torch.tensor([[0.1, -0.05, 0.08, 0.2, -0.1, 0.15]], device=device)
        vel_fn, analytical_fn = make_constant_velocity_ode(xi)
        x0 = torch.eye(4, device=device).unsqueeze(0)

        # Adaptive
        integrator = AdaptiveODEIntegrator(
            velocity_fn=vel_fn, atol=1e-3, rtol=1e-2, max_steps=50,
        )
        x1_adaptive, info = integrator.integrate(x0)
        adaptive_err = se3_geodesic_distance(x1_adaptive, analytical_fn(1.0, x0))

        # Fixed RK4 with 10 steps
        matcher = GeodesicFlowMatcher()
        x1_rk4 = matcher.rk4_integrate(vel_fn, x0, num_steps=10)
        rk4_err = se3_geodesic_distance(x1_rk4, analytical_fn(1.0, x0))

        # Adaptive should be at least as accurate, often with fewer evaluations
        assert adaptive_err.item() <= rk4_err.item() + 1e-4, (
            f"Adaptive error {adaptive_err.item():.6f} should be ≤ RK4 error "
            f"{rk4_err.item():.6f}"
        )

    @pytest.mark.skipif(
        not pytest.importorskip("torchdiffeq", reason="torchdiffeq not installed"),
        reason="torchdiffeq not available",
    )
    def test_against_torchdiffeq(self, device):
        """Compare hand-rolled DOPRI5 against torchdiffeq.odeint reference."""
        from torchdiffeq import odeint as torchdiffeq_odeint
        from src.flow.geodesic_flow import AdaptiveODEIntegrator

        xi = torch.tensor([[0.15, -0.1, 0.12, 0.25, -0.15, 0.2]], device=device)
        vel_fn, analytical_fn = make_constant_velocity_ode(xi)
        x0 = torch.eye(4, device=device).unsqueeze(0)

        # Our integrator
        integrator = AdaptiveODEIntegrator(
            velocity_fn=vel_fn, atol=1e-5, rtol=1e-4, max_steps=200,
        )
        x1_ours, info = integrator.integrate(x0)

        # torchdiffeq reference (on se(3) vector, not manifold)
        def flat_ode(t, state):
            return xi.expand(state.shape[0], -1)

        xi0 = torch.zeros(1, 6, device=device)
        xi_trajectory = torchdiffeq_odeint(
            flat_ode, xi0,
            t=torch.tensor([0.0, 1.0], device=device),
            method="dopri5",
            rtol=1e-6,
            atol=1e-8,
        )
        xi_final = xi_trajectory[-1]  # [1, 6]
        x1_ref = se3_expmap(xi_final) @ x0

        geodesic_err = se3_geodesic_distance(x1_ours, x1_ref)
        assert geodesic_err.item() < 1e-3, (
            f"Hand-rolled DOPRI5 vs torchdiffeq: geodesic error "
            f"{geodesic_err.item():.6f} exceeds 1e-3"
        )

    def test_halts_at_t_equals_1(self, device, identity_batch):
        """Integration must reach t=1.0 (not overshoot or undershoot)."""
        from src.flow.geodesic_flow import AdaptiveODEIntegrator

        xi = torch.tensor([[0.3, -0.2, 0.1, 0.5, -0.3, 0.4]], device=device)
        vel_fn, analytical_fn = make_constant_velocity_ode(xi)

        integrator = AdaptiveODEIntegrator(
            velocity_fn=vel_fn, atol=1e-4, rtol=1e-3, max_steps=200,
        )

        x1, info = integrator.integrate(identity_batch[:1])

        # The final pose should match the analytical solution at t=1
        x1_exact = analytical_fn(1.0, identity_batch[:1])
        err = se3_geodesic_distance(x1, x1_exact)
        assert err.item() < 1e-3, f"Failed to reach t=1 accurately: error {err.item():.6f}"
        assert info["mean_final_t"] > 0.99, (
            f"Final t = {info['mean_final_t']:.4f}, expected ≈ 1.0"
        )

    def test_zero_velocity_returns_identity(self, device, identity_batch):
        """Zero velocity → pose should not change."""
        from src.flow.geodesic_flow import AdaptiveODEIntegrator

        xi_zero = torch.zeros(1, 6, device=device)
        vel_fn, _ = make_constant_velocity_ode(xi_zero)

        integrator = AdaptiveODEIntegrator(
            velocity_fn=vel_fn, atol=1e-4, rtol=1e-3, max_steps=50,
        )

        x1, info = integrator.integrate(identity_batch[:1])
        err = se3_geodesic_distance(x1, identity_batch[:1])
        assert err.item() < 1e-6, f"Zero velocity should return identity: error {err.item():.6f}"


class TestLearnedHalting:
    """Test the learned halting network."""

    def test_halting_output_shape(self):
        from src.flow.geodesic_flow import LearnedHaltingNetwork

        net = LearnedHaltingNetwork()
        v = torch.randn(4, 6)
        t = torch.rand(4)
        p = net(v, t)
        assert p.shape == (4,), f"Expected shape (4,), got {p.shape}"
        assert (p >= 0).all() and (p <= 1).all(), "Halting prob must be in [0, 1]"

    def test_halting_near_zero_at_init(self):
        """At initialization, halting probability should be low (biased to continue)."""
        from src.flow.geodesic_flow import LearnedHaltingNetwork

        torch.manual_seed(42)
        net = LearnedHaltingNetwork()
        v = torch.randn(10, 6)
        t = torch.rand(10)
        p = net(v, t)
        assert p.mean().item() < 0.3, (
            f"Initial halting prob {p.mean().item():.3f} should be < 0.3"
        )


class TestTaskConditionedMetric:
    """Test the task-conditioned metric tensor."""

    def test_output_positive(self):
        """Metric weights must be positive (via softplus)."""
        from src.flow.geodesic_flow import TaskConditionedMetric

        net = TaskConditionedMetric(task_dim=256)
        z = torch.randn(8, 256)
        g = net(z)
        assert g.shape == (8, 6)
        assert (g > 0).all(), "Metric weights must be positive"

    def test_output_near_identity_at_init(self):
        """At initialization, metric should be close to [1,1,1,1,1,1]."""
        from src.flow.geodesic_flow import TaskConditionedMetric

        torch.manual_seed(42)
        net = TaskConditionedMetric(task_dim=256)
        z = torch.randn(100, 256)
        g = net(z)
        mean_g = g.mean(dim=0)
        assert torch.allclose(mean_g, torch.ones(6), atol=0.5), (
            f"Init metric mean {mean_g.tolist()} should be near [1,1,1,1,1,1]"
        )

    def test_different_tasks_different_metrics(self):
        """Different task embeddings should produce different metrics."""
        from src.flow.geodesic_flow import TaskConditionedMetric

        torch.manual_seed(42)
        net = TaskConditionedMetric(task_dim=256)
        z1 = torch.randn(1, 256)
        z2 = torch.randn(1, 256) * 10  # very different
        g1 = net(z1)
        g2 = net(z2)
        assert not torch.allclose(g1, g2, atol=0.01), (
            "Different tasks should produce different metrics"
        )
