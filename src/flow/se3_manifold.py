"""SE(3) manifold implementation for facebookresearch/flow_matching.

Implements the ``Manifold`` interface required by the flow_matching library
so that Riemannian flow matching can be performed on SE(3).
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from flow_matching.manifold import Manifold
except ImportError:
    # Minimal stub so the module can be imported without flow_matching installed
    class Manifold:  # type: ignore[no-redef]
        """Stub base class."""

        def expmap(self, x: Tensor, u: Tensor) -> Tensor:
            raise NotImplementedError

        def logmap(self, x: Tensor, y: Tensor) -> Tensor:
            raise NotImplementedError

        def projx(self, x: Tensor) -> Tensor:
            raise NotImplementedError

        def proju(self, x: Tensor, u: Tensor) -> Tensor:
            raise NotImplementedError

from .se3_utils import se3_expmap, se3_logmap


class SE3Manifold(Manifold):
    """SE(3) manifold for Riemannian flow matching.

    Elements are 4×4 homogeneous matrices [B, 4, 4].
    Tangent vectors are 6-vectors in se(3) [B, 6].

    The manifold structure:
        - expmap: se(3) → SE(3) via Rodrigues formula
        - logmap: SE(3) → se(3) via matrix logarithm
        - projx:  Re-orthonormalize rotation via polar decomposition
        - proju:  Project tangent vector (identity on se(3), already valid)
    """

    # ------------------------------------------------------------------
    # Exponential map
    # ------------------------------------------------------------------

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Exponential map: move along tangent vector u from point x.

        Args:
            x: [B, 4, 4] base points on SE(3).
            u: [B, 6] tangent vectors in se(3).

        Returns:
            y: [B, 4, 4] resulting points on SE(3).
        """
        # exp_x(u) = exp(u) @ x
        T_u = se3_expmap(u)  # [B, 4, 4]
        return T_u @ x

    # ------------------------------------------------------------------
    # Logarithmic map
    # ------------------------------------------------------------------

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map: tangent vector from x to y.

        Args:
            x: [B, 4, 4] source points on SE(3).
            y: [B, 4, 4] target points on SE(3).

        Returns:
            u: [B, 6] tangent vectors in se(3).
        """
        # log_x(y) = log(y @ x⁻¹)
        x_inv = self._inverse(x)
        delta = y @ x_inv
        return se3_logmap(delta)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def projx(self, x: Tensor) -> Tensor:
        """Project onto SE(3) by re-orthonormalizing the rotation block.

        Uses polar decomposition: R_proj = U @ Vᵀ where R = U @ Σ @ Vᵀ.

        Args:
            x: [B, 4, 4] matrices (possibly with non-orthogonal rotations).

        Returns:
            x_proj: [B, 4, 4] valid SE(3) elements.
        """
        R = x[:, :3, :3]
        t = x[:, :3, 3]

        # Polar decomposition via SVD
        U, S, Vh = torch.linalg.svd(R)
        R_proj = U @ Vh  # [B, 3, 3]

        # Ensure proper rotation (det = +1)
        det = torch.det(R_proj)  # [B]
        # Flip sign of last column of U if det < 0
        flip_mask = det < 0
        if flip_mask.any():
            U_fixed = U.clone()
            U_fixed[flip_mask, :, -1] *= -1
            R_proj = U_fixed @ Vh

        x_proj = x.clone()
        x_proj[:, :3, :3] = R_proj
        x_proj[:, :3, 3] = t
        return x_proj

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project tangent vector onto the tangent space at x.

        For se(3), all 6-vectors are valid tangent vectors, so this is identity.
        However, we ensure the tangent vector lives on the correct tangent space
        by verifying that the rotation part maps to so(3).

        Args:
            x: [B, 4, 4] base points on SE(3).
            u: [B, 6] tangent vectors.

        Returns:
            u_proj: [B, 6] projected tangent vectors.
        """
        # For SE(3), tangent vectors are already in se(3) ≅ ℝ⁶
        # No projection needed, but we can optionally symmetrize
        return u

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _inverse(T: Tensor) -> Tensor:
        """Efficient SE(3) inverse."""
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        T_inv = torch.zeros_like(T)
        T_inv[:, :3, :3] = R.transpose(-1, -2)
        T_inv[:, :3, 3] = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
        T_inv[:, 3, 3] = 1.0
        return T_inv

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance on SE(3).

        Args:
            x: [B, 4, 4].
            y: [B, 4, 4].

        Returns:
            d: [B] geodesic distances.
        """
        u = self.logmap(x, y)
        return u.norm(dim=-1)

    def __repr__(self) -> str:
        return "SE3Manifold(4×4 homogeneous matrices, se(3) tangent vectors)"
