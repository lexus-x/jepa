"""SE(3) Lie group utilities: exponential map, logarithmic map, geodesic interpolation.

All operations are batched PyTorch, fully differentiable, and use 4×4
homogeneous matrix representation for SE(3) elements.

Convention:
    T ∈ SE(3) is represented as a 4×4 matrix::

        T = [[R, t],
             [0, 1]]

    where R ∈ SO(3) is a 3×3 rotation matrix and t ∈ ℝ³ is a translation.

    The Lie algebra se(3) is parameterized by a 6-vector ξ = [ω, υ] where
    ω ∈ ℝ³ is angular velocity and υ ∈ ℝ³ is linear velocity.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ======================================================================
# Small-angle threshold for Taylor expansion
# ======================================================================
_SMALL_ANGLE: float = 1e-6


# ======================================================================
# SO(3) helpers
# ======================================================================

def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Build batch of 3×3 skew-symmetric matrices from [B, 3] vectors.

    [v]× = [[ 0,   -v_z,  v_y],
            [ v_z,  0,   -v_x],
            [-v_y,  v_x,  0  ]]
    """
    B = v.shape[0]
    device, dtype = v.device, v.dtype
    O = torch.zeros(B, device=device, dtype=dtype)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack([
        O,   -vz,  vy,
        vz,  O,   -vx,
        -vy, vx,  O,
    ], dim=-1).reshape(B, 3, 3)


def _so3_expmap(omega: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential map via Rodrigues formula.

    Args:
        omega: [B, 3] axis-angle vectors.

    Returns:
        Rotation matrices [B, 3, 3].
    """
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [B, 1]
    theta = theta_sq.sqrt()  # [B, 1]

    # Normalized axis (safe for theta → 0)
    axis = omega / (theta + _SMALL_ANGLE)  # [B, 3]

    K = _skew_symmetric(axis)  # [B, 3, 3]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)

    # Rodrigues: R = I + sin(θ) K + (1 - cos(θ)) K²
    sin_theta = theta.unsqueeze(-1).sin()  # [B, 1, 1]
    cos_theta = theta.unsqueeze(-1).cos()  # [B, 1, 1]

    R = I + sin_theta * K + (1.0 - cos_theta) * (K @ K)

    # For very small angles, use first-order approximation
    small = (theta_sq.squeeze(-1) < _SMALL_ANGLE)  # [B]
    if small.any():
        R_small = I + K  # First-order: R ≈ I + [ω]×
        R = torch.where(small.view(-1, 1, 1), R_small, R)

    return R


def _so3_logmap(R: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithmic map.

    Args:
        R: [B, 3, 3] rotation matrices.

    Returns:
        Axis-angle vectors [B, 3].
    """
    # cos(θ) = (tr(R) - 1) / 2
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # [B]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = cos_theta.acos()  # [B]

    # Axis from skew-symmetric part: ω = θ / (2 sin θ) * (R - Rᵀ)∨
    R_minus_RT = R - R.transpose(-1, -2)  # [B, 3, 3]
    # Extract the 3 independent components: [R32, R13, R21]
    axis_unnorm = torch.stack([
        R_minus_RT[:, 2, 1],
        R_minus_RT[:, 0, 2],
        R_minus_RT[:, 1, 0],
    ], dim=-1)  # [B, 3]

    sin_theta = theta.sin()
    # Scale factor: θ / (2 sin θ)
    scale = theta / (2.0 * sin_theta + _SMALL_ANGLE)
    omega = scale.unsqueeze(-1) * axis_unnorm  # [B, 3]

    # Handle small angles: ω ≈ (R - Rᵀ)∨ / 2
    small = theta.abs() < _SMALL_ANGLE
    if small.any():
        omega_small = axis_unnorm / 2.0
        omega = torch.where(small.unsqueeze(-1), omega_small, omega)

    # Handle θ ≈ π (180°): special case
    near_pi = (theta - math.pi).abs() < 1e-4
    if near_pi.any():
        # For θ ≈ π, extract axis from R + I
        R_plus_I = R + I_like(R)
        # Axis is the column of R+I with the largest norm
        col_norms = R_plus_I.norm(dim=-1)  # [B, 3]
        best_col = col_norms.argmax(dim=-1)  # [B]
        batch_idx = torch.arange(R.shape[0], device=R.device)
        axis_pi = R_plus_I[batch_idx, :, best_col]  # [B, 3]
        axis_pi = F.normalize(axis_pi, dim=-1)
        omega_pi = math.pi * axis_pi
        omega = torch.where(near_pi.unsqueeze(-1), omega_pi, omega)

    return omega


def I_like(R: torch.Tensor) -> torch.Tensor:
    """Identity matrix matching batch size and dtype of R."""
    return torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], -1, -1)


# ======================================================================
# SE(3) exponential map
# ======================================================================

def se3_expmap(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential map: se(3) → SE(3).

    Uses the closed-form Rodrigues formula for SE(3).

    Args:
        xi: [B, 6] Lie algebra elements, concatenation [ω, υ] where
            ω ∈ ℝ³ (angular) and υ ∈ ℝ³ (linear).

    Returns:
        T: [B, 4, 4] homogeneous transformation matrices.
    """
    B = xi.shape[0]
    device, dtype = xi.device, xi.dtype

    omega = xi[:, :3]  # [B, 3]
    v = xi[:, 3:]  # [B, 3]

    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [B, 1]
    theta = theta_sq.sqrt()  # [B, 1]
    theta_4 = theta_sq * theta_sq  # θ⁴

    R = _so3_expmap(omega)  # [B, 3, 3]

    # V matrix: maps linear velocity to translation
    # V = I + (1-cosθ)/θ² [ω]× + (θ-sinθ)/θ³ [ω]×²
    K = _skew_symmetric(omega)  # [B, 3, 3]
    I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

    sin_theta = theta.unsqueeze(-1).sin()
    cos_theta = theta.unsqueeze(-1).cos()

    # Coefficients with safe division
    theta_sq_3d = theta_sq.unsqueeze(-1)  # [B, 1, 1]
    theta_3d = theta.unsqueeze(-1)  # [B, 1, 1]
    theta_4_3d = theta_4.unsqueeze(-1)  # [B, 1, 1]

    c1 = (1.0 - cos_theta) / (theta_sq_3d + _SMALL_ANGLE)
    c2 = (theta_3d - sin_theta) / (theta_4_3d / (theta_3d + _SMALL_ANGLE) + _SMALL_ANGLE)

    V = I3 + c1 * K + c2 * (K @ K)

    # For small angles, V ≈ I + K/2
    small = (theta_sq.squeeze(-1) < _SMALL_ANGLE)
    if small.any():
        V_small = I3 + K / 2.0
        V = torch.where(small.view(-1, 1, 1), V_small, V)

    # Translation: t = V @ v
    t = (V @ v.unsqueeze(-1)).squeeze(-1)  # [B, 3]

    # Assemble 4×4 matrix
    T = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T


# ======================================================================
# SE(3) logarithmic map
# ======================================================================

def se3_logmap(T: torch.Tensor) -> torch.Tensor:
    """SE(3) logarithmic map: SE(3) → se(3).

    Args:
        T: [B, 4, 4] homogeneous transformation matrices.

    Returns:
        xi: [B, 6] Lie algebra elements [ω, υ].
    """
    R = T[:, :3, :3]  # [B, 3, 3]
    t = T[:, :3, 3]  # [B, 3]

    omega = _so3_logmap(R)  # [B, 3]

    # V⁻¹: inverse of the V matrix from expmap
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [B, 1]
    theta = theta_sq.sqrt()  # [B, 1]

    K = _skew_symmetric(omega)  # [B, 3, 3]
    I3 = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)

    sin_theta = theta.unsqueeze(-1).sin()
    cos_theta = theta.unsqueeze(-1).cos()
    theta_3d = theta.unsqueeze(-1)
    theta_sq_3d = theta_sq.unsqueeze(-1)

    # V⁻¹ = I - 0.5 K + (1/θ² - (1+cosθ)/(2θ sinθ)) K²
    half_theta = theta_3d / 2.0
    c1 = -0.5 * torch.ones_like(theta_sq_3d)
    c2 = (1.0 / (theta_sq_3d + _SMALL_ANGLE) -
          (1.0 + cos_theta) / (2.0 * theta_3d * sin_theta + _SMALL_ANGLE))

    V_inv = I3 + c1 * K + c2 * (K @ K)

    # Small angle fallback: V⁻¹ ≈ I - K/2
    small = (theta_sq.squeeze(-1) < _SMALL_ANGLE)
    if small.any():
        V_inv_small = I3 - K / 2.0
        V_inv = torch.where(small.view(-1, 1, 1), V_inv_small, V_inv)

    # Linear velocity: υ = V⁻¹ @ t
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)  # [B, 3]

    return torch.cat([omega, v], dim=-1)  # [B, 6]


# ======================================================================
# Geodesic interpolation
# ======================================================================

def geodesic_interpolation(
    T0: torch.Tensor,
    T1: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Geodesic interpolation on SE(3): γ(t) = exp(t · log(T₁ T₀⁻¹)) T₀.

    Args:
        T0: [B, 4, 4] start poses.
        T1: [B, 4, 4] end poses.
        t: Interpolation parameter in [0, 1]. Can be scalar or [B].

    Returns:
        T_t: [B, 4, 4] interpolated poses.
    """
    # Relative transform: ΔT = T1 @ T0⁻¹
    T0_inv = _se3_inverse(T0)
    delta_T = T1 @ T0_inv  # [B, 4, 4]

    # Log map: ξ = log(ΔT)
    xi = se3_logmap(delta_T)  # [B, 6]

    # Scale by t
    if isinstance(t, (int, float)):
        xi_t = xi * t
    else:
        xi_t = xi * t.unsqueeze(-1)

    # Exponential map and compose
    T_t = se3_expmap(xi_t) @ T0  # [B, 4, 4]
    return T_t


# ======================================================================
# Wrapped Gaussian on SE(3)
# ======================================================================

def wrapped_gaussian(
    mean: torch.Tensor,
    sigma_trans: float = 0.1,
    sigma_rot: float = 0.3,
    num_samples: int = 1,
) -> torch.Tensor:
    """Sample from a wrapped Gaussian distribution on SE(3).

    The wrapped Gaussian is defined as:
        x = exp(ξ) @ mean,  where ξ ~ N(0, diag(σ_rot², σ_rot², σ_rot², σ_trans², σ_trans², σ_trans²))

    Args:
        mean: [B, 4, 4] mean poses.
        sigma_trans: Standard deviation for translation components.
        sigma_rot: Standard deviation for rotation components.
        num_samples: Number of samples per mean.

    Returns:
        samples: [B, num_samples, 4, 4] sampled poses.
    """
    B = mean.shape[0]
    device, dtype = mean.device, mean.dtype

    # Sample noise in se(3)
    sigma = torch.tensor(
        [sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans],
        device=device, dtype=dtype,
    )
    noise = torch.randn(B, num_samples, 6, device=device, dtype=dtype) * sigma

    # Apply exponential map: [B, num_samples, 6] → [B*num_samples, 6] → [B*num_samples, 4, 4]
    noise_flat = noise.reshape(B * num_samples, 6)
    T_noise = se3_expmap(noise_flat)  # [B*num_samples, 4, 4]

    # Compose with mean
    mean_expanded = mean.unsqueeze(1).expand(B, num_samples, 4, 4)
    mean_flat = mean_expanded.reshape(B * num_samples, 4, 4)
    samples = T_noise @ mean_flat  # [B*num_samples, 4, 4]

    return samples.reshape(B, num_samples, 4, 4)


# ======================================================================
# Geodesic distance
# ======================================================================

def se3_geodesic_distance(T0: torch.Tensor, T1: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on SE(3): d(T0, T1) = ‖log(T0⁻¹ T1)‖.

    Args:
        T0: [B, 4, 4].
        T1: [B, 4, 4].

    Returns:
        distances: [B] geodesic distances.
    """
    T0_inv = _se3_inverse(T0)
    delta = T0_inv @ T1
    xi = se3_logmap(delta)  # [B, 6]
    return xi.norm(dim=-1)  # [B]


# ======================================================================
# Inverse helper
# ======================================================================

def _se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Efficient SE(3) inverse: T⁻¹ = [[Rᵀ, -Rᵀt], [0, 1]].

    Args:
        T: [B, 4, 4].

    Returns:
        T_inv: [B, 4, 4].
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3]

    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = R.transpose(-1, -2)
    T_inv[:, :3, 3] = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    T_inv[:, 3, 3] = 1.0
    return T_inv


# ======================================================================
# Quaternion ↔ Matrix conversion (utility)
# ======================================================================

def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions [w, x, y, z] to rotation matrices.

    Args:
        q: [B, 4] unit quaternions.

    Returns:
        R: [B, 3, 3] rotation matrices.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def rotation_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to unit quaternions [w, x, y, z].

    Uses Shepperd's method for numerical stability.

    Args:
        R: [B, 3, 3] rotation matrices.

    Returns:
        q: [B, 4] unit quaternions.
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # [B]

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s1 = (trace + 1.0).sqrt() * 2  # 4w
    w1 = s1 / 4
    x1 = (R[:, 2, 1] - R[:, 1, 2]) / s1
    y1 = (R[:, 0, 2] - R[:, 2, 0]) / s1
    z1 = (R[:, 1, 0] - R[:, 0, 1]) / s1

    # Case 2: R[0,0] is largest
    s2 = (1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).sqrt() * 2
    w2 = (R[:, 2, 1] - R[:, 1, 2]) / s2
    x2 = s2 / 4
    y2 = (R[:, 0, 1] + R[:, 1, 0]) / s2
    z2 = (R[:, 0, 2] + R[:, 2, 0]) / s2

    # Case 3: R[1,1] is largest
    s3 = (1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).sqrt() * 2
    w3 = (R[:, 0, 2] - R[:, 2, 0]) / s3
    x3 = (R[:, 0, 1] + R[:, 1, 0]) / s3
    y3 = s3 / 4
    z3 = (R[:, 1, 2] + R[:, 2, 1]) / s3

    # Case 4: R[2,2] is largest
    s4 = (1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).sqrt() * 2
    w4 = (R[:, 1, 0] - R[:, 0, 1]) / s4
    x4 = (R[:, 0, 2] + R[:, 2, 0]) / s4
    y4 = (R[:, 1, 2] + R[:, 2, 1]) / s4
    z4 = s4 / 4

    mask1 = trace > 0
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    mask4 = ~mask1 & ~mask2 & ~mask3

    for i, (w, x, y, z) in enumerate([
        (w1, x1, y1, z1), (w2, x2, y2, z2),
        (w3, x3, y3, z3), (w4, x4, y4, z4),
    ]):
        mask = [mask1, mask2, mask3, mask4][i]
        if mask.any():
            q[mask, 0] = w[mask]
            q[mask, 1] = x[mask]
            q[mask, 2] = y[mask]
            q[mask, 3] = z[mask]

    return F.normalize(q, dim=-1)
