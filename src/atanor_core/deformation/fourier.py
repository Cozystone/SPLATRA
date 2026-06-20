"""MLP-free Fourier + polynomial morphing (Gaussian-Flow / DDDM style).

No neural network and no inference. Position over time is a closed-form
spectral series::

    pos(t) = origin
           + Σ_n poly[:, :, n] · t^n              # smooth polynomial drift
           + anneal(t) · Σ_k fourier·basis_k(t)   # annealed "smoke" turbulence

where ``basis_k = [sin(2πk t / T), cos(2πk t / T)]`` and
``anneal(t) = max(0, 1 - t / T)``.

The polynomial linear term carries the bulk motion from ``origin`` to
``target`` (at ``t = T`` the linear term equals ``target - origin``), so the
field lands exactly on the target. The Fourier term injects turbulence that
anneals to zero by ``t = T`` — the cloud "settles". Computing ``Δa(t)`` is a
matrix multiply, so it is sub-millisecond.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.sgf import DeformationCoeffs


class FourierDeformer:
    """Stateful spectral deformer over a fixed Gaussian count.

    Args:
        means: [N, 3] initial positions (origin == target until a morph fires).
        period: morph period T.
        n_p: number of polynomial terms (>= 2 to carry a linear term).
        n_f: number of Fourier harmonics.
        seed: RNG seed for reproducible turbulence.
    """

    def __init__(
        self,
        means: np.ndarray,
        period: float = 1.0,
        n_p: int = 3,
        n_f: int = 8,
        seed: int = 0,
    ) -> None:
        means = np.asarray(means, dtype=np.float32)
        n = means.shape[0]
        self.coeffs = DeformationCoeffs.zeros(n, n_p=n_p, n_f=n_f, period=period)
        self.coeffs.origin = means.copy()
        self.coeffs.target = means.copy()
        self.period = float(period)
        self.t = 0.0
        self._pos = means.copy()
        self._active = False
        self._rng = np.random.default_rng(seed)

    @property
    def positions(self) -> np.ndarray:
        return self._pos

    @property
    def done(self) -> bool:
        """True when no morph is in progress (settled on target / static)."""
        return not self._active

    def trigger_morph(self, new_positions: np.ndarray, turbulence: float = 0.0) -> None:
        """Begin a morph from the current positions toward ``new_positions``.

        Sets the polynomial linear term to ``target - origin`` and seeds the
        Fourier coefficients with Gaussian turbulence of the given std-dev.
        Resets ``t`` to 0.
        """
        new_positions = np.asarray(new_positions, dtype=np.float32)
        self.coeffs.origin = self._pos.copy()
        self.coeffs.target = new_positions.copy()
        self.coeffs.poly[:] = 0.0
        # Linear term carries origin -> target over [0, T] (t^1, evaluated at T).
        self.coeffs.poly[:, :, 1] = self.coeffs.target - self.coeffs.origin
        # Random-phase turbulence; annealed to zero by t = T.
        self.coeffs.fourier = self._rng.normal(
            0.0, float(turbulence), size=self.coeffs.fourier.shape
        ).astype(np.float32)
        self.t = 0.0
        self._active = True

    def step(self, t: Optional[float] = None, dt: float = 0.033) -> np.ndarray:
        """Advance the morph and return positions [N, 3].

        If ``t`` is None the internal clock advances by ``dt``; otherwise the
        clock is set to ``t``. Once ``t >= T`` the field snaps to ``target``
        and the morph is marked done.
        """
        if t is None:
            self.t += float(dt)
        else:
            self.t = float(t)

        T = self.period
        if self.t >= T:
            self.t = T
            self._pos = self.coeffs.target.copy()
            self._active = False
            return self._pos

        tt = self.t
        # Polynomial drift: Σ_n poly[:, :, n] · t^n.
        powers = tt ** np.arange(self.coeffs.n_p, dtype=np.float32)  # [n_p]
        d_poly = self.coeffs.poly @ powers  # [N, 3, n_p] @ [n_p] -> [N, 3]

        # Fourier turbulence basis [sin_1..sin_nf, cos_1..cos_nf].
        ks = np.arange(1, self.coeffs.n_f + 1, dtype=np.float32)
        ang = 2.0 * np.pi * ks * tt / T
        basis = np.concatenate([np.sin(ang), np.cos(ang)]).astype(np.float32)  # [2*n_f]
        anneal = max(0.0, 1.0 - tt / T)
        d_four = (self.coeffs.fourier @ basis) * anneal  # [N, 3]

        self._pos = (self.coeffs.origin + d_poly + d_four).astype(np.float32)
        return self._pos
