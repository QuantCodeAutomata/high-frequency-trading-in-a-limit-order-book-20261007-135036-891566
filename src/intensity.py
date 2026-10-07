"""
Execution intensity models for the Avellaneda & Stoikov (2008) framework.

Section 2.5 "The trading intensity" (pp. 6–7):

  The Poisson intensity at which a limit order at distance δ from the
  mid-price is executed is modelled as λ(δ) = Λ·P(Δp > δ), where Δp
  is the temporary price impact of an incoming market order.

Two cases are derived from a power-law market-order size density
f^Q(x) ∝ x^{-1-α} (eq. 2.8):

  Case A – logarithmic impact Δp ∝ ln(Q) (Potters & Bouchaud):
      λ(δ) = A·exp(−k·δ),  A = Λ/α,  k = α·K   (eq. 2.11)

  Case B – power-law impact Δp ∝ Q^β:
      λ(δ) = B·δ^{−α/β}

The exponential form is the preferred operational model because it
yields analytically tractable quote formulas (Sections 3.1–3.2).
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class IntensityModel(ABC):
    """Abstract base for execution intensity models."""

    @abstractmethod
    def intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Return execution intensity λ(δ) for quote distance δ ≥ 0.

        Parameters
        ----------
        delta:
            Distance of the limit-order quote from the mid-price (≥ 0).

        Returns
        -------
        λ(δ): intensity (events per unit time).
        """

    @abstractmethod
    def d_intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Return dλ/dδ for use in first-order optimality conditions."""


class ExponentialIntensity(IntensityModel):
    """
    Exponential execution-intensity model (Section 2.5, eq. 2.11).

    λ(δ) = A · exp(−k · δ)

    Derived from power-law market-order size density f^Q(x) ∝ x^{-1-α}
    combined with logarithmic temporary price impact Δp = (1/K)·ln(Q):
      A = Λ/α,  k = α·K.

    Under this model the ratio λ/λ′ = −1/k (since λ′ = −k·λ),
    which simplifies the first-order conditions (eqs. 3.6–3.7).
    """

    def __init__(self, A: float, k: float) -> None:
        """
        Parameters
        ----------
        A:
            Intensity scale (≡ Λ/α from calibration eq. 2.11).
        k:
            Exponential decay rate (≡ α·K from calibration eq. 2.11).
        """
        if A <= 0:
            raise ValueError(f"A must be positive, got {A}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.A = float(A)
        self.k = float(k)

    def intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """λ(δ) = A · exp(−k · δ)."""
        return self.A * np.exp(-self.k * delta)

    def d_intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """dλ/dδ = −k · λ(δ)."""
        return -self.k * self.intensity(delta)

    @classmethod
    def from_micro(cls, Lambda: float, alpha: float, K: float) -> "ExponentialIntensity":
        """
        Calibrate from microstructure primitives (eq. 2.11).

        Parameters
        ----------
        Lambda:
            Overall market-order arrival frequency.
        alpha:
            Tail exponent of market-order size distribution (eq. 2.8).
        K:
            Proportionality constant in Δp = (1/K)·ln(Q).
        """
        A = Lambda / alpha
        k = alpha * K
        return cls(A=A, k=k)


class PowerLawIntensity(IntensityModel):
    """
    Power-law execution-intensity model (Section 2.5).

    λ(δ) = B · δ^{−α/β}

    Derived from power-law size density and power-law price impact
    Δp ∝ Q^β (eq. 2.9).  Only valid for δ > 0.
    """

    def __init__(self, B: float, alpha: float, beta: float) -> None:
        """
        Parameters
        ----------
        B:
            Scale constant.
        alpha:
            Tail exponent of market-order size distribution.
        beta:
            Price-impact exponent (Δp ∝ Q^β).
        """
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.B = float(B)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._exponent = alpha / beta

    def intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """λ(δ) = B · δ^{−α/β}  (δ > 0)."""
        delta = np.asarray(delta, dtype=float)
        if np.any(delta <= 0):
            raise ValueError("PowerLawIntensity requires δ > 0")
        return self.B * delta ** (-self._exponent)

    def d_intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """dλ/dδ = −(α/β) · B · δ^{−α/β − 1}."""
        delta = np.asarray(delta, dtype=float)
        if np.any(delta <= 0):
            raise ValueError("PowerLawIntensity requires δ > 0")
        return -self._exponent * self.B * delta ** (-self._exponent - 1)


def compare_intensities(
    delta_grid: np.ndarray,
    exp_model: ExponentialIntensity,
    pl_model: PowerLawIntensity,
) -> dict:
    """
    Evaluate both intensity models on a common δ grid for comparison.

    Parameters
    ----------
    delta_grid:
        Array of quote distances (> 0) at which to evaluate.
    exp_model:
        Calibrated ExponentialIntensity instance.
    pl_model:
        Calibrated PowerLawIntensity instance.

    Returns
    -------
    dict with keys 'delta', 'lambda_exp', 'lambda_pl'.
    """
    return {
        "delta": delta_grid,
        "lambda_exp": exp_model.intensity(delta_grid),
        "lambda_pl": pl_model.intensity(delta_grid),
    }
