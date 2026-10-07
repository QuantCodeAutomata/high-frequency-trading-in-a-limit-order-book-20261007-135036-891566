"""
Reservation (indifference) price computations for Avellaneda & Stoikov (2008).

Implements Sections 2.2 and 2.3 for an inactive agent (frozen inventory)
under arithmetic Brownian mid-price dynamics (eq. 2.1) and exponential utility.

Finite-horizon formulas (Section 2.2, eqs. 2.3–2.7):
  Value function:
    v(x,s,q,t) = −exp(−γx) · exp(−γqs) · exp(γ²q²σ²(T−t)/2)   (2.3)

  Reservation ask and bid prices (Definition 1, eqs. 2.6–2.7):
    r^a(s,q,t) = s + (1 − 2q) · γσ²(T−t)/2
    r^b(s,q,t) = s + (−1 − 2q) · γσ²(T−t)/2

  Average reservation (indifference) price:
    r(s,q,t) = s − q · γσ²(T−t)

Infinite-horizon formulas (Section 2.3):
  Objective: Ē[∫_0^∞ −exp(−ωt) exp(−γ(x+qS_t)) dt]
  Stationary reservation prices:
    r̄^a(s,q) = s + (1/γ) ln(1 + (1−2q)γ²σ² / (2ω − γ²q²σ²))
    r̄^b(s,q) = s + (1/γ) ln(1 + (−1−2q)γ²σ² / (2ω − γ²q²σ²))
  Admissibility: ω > (1/2)γ²σ²q²
  Bounded choice: ω = (1/2)γ²σ²(q_max+1)²
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Finite-horizon inactive-agent formulas (Section 2.2)
# ---------------------------------------------------------------------------

def finite_value(
    x: float,
    s: float,
    q: float | int,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Value function for a frozen-inventory agent (eq. 2.3).

    v(x,s,q,t) = −exp(−γx) · exp(−γqs) · exp(γ²q²σ²(T−t)/2)

    Parameters
    ----------
    x:     Cash wealth.
    s:     Current mid-price.
    q:     Inventory (integer shares).
    t:     Current time.
    T:     Terminal time.
    sigma: Mid-price volatility (arithmetic Brownian, eq. 2.1).
    gamma: Absolute risk-aversion coefficient.
    """
    tau = T - t
    assert tau >= 0, f"t={t} must not exceed T={T}"
    return (
        -np.exp(-gamma * x)
        * np.exp(-gamma * q * s)
        * np.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
    )


def finite_reservation_ask(
    s: float | np.ndarray,
    q: float | int | np.ndarray,
    t: float | np.ndarray,
    T: float,
    sigma: float,
    gamma: float,
) -> float | np.ndarray:
    """
    Reservation ask price for a frozen-inventory agent (eq. 2.6).

    r^a(s,q,t) = s + (1 − 2q) · γσ²(T−t) / 2

    The agent is willing to sell one share at this price.
    """
    tau = T - t
    return s + (1.0 - 2.0 * q) * gamma * sigma**2 * tau / 2.0


def finite_reservation_bid(
    s: float | np.ndarray,
    q: float | int | np.ndarray,
    t: float | np.ndarray,
    T: float,
    sigma: float,
    gamma: float,
) -> float | np.ndarray:
    """
    Reservation bid price for a frozen-inventory agent (eq. 2.7).

    r^b(s,q,t) = s + (−1 − 2q) · γσ²(T−t) / 2

    The agent is willing to buy one share at this price.
    """
    tau = T - t
    return s + (-1.0 - 2.0 * q) * gamma * sigma**2 * tau / 2.0


def finite_reservation_mid(
    s: float | np.ndarray,
    q: float | int | np.ndarray,
    t: float | np.ndarray,
    T: float,
    sigma: float,
    gamma: float,
) -> float | np.ndarray:
    """
    Average (indifference) reservation price for the finite-horizon agent.

    r(s,q,t) = (r^a + r^b)/2 = s − q · γσ²(T−t)

    This is the key quantity used in the simulation (eq. 3.17):
    positive inventory lowers r below s; negative inventory raises r above s.
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Infinite-horizon stationary formulas (Section 2.3)
# ---------------------------------------------------------------------------

def omega_bounded(
    q_max: int,
    gamma: float,
    sigma: float,
) -> float:
    """
    Natural choice of discount parameter ensuring bounded reservation prices.

    ω = (1/2) · γ² · σ² · (q_max + 1)²   (Section 2.3)

    This satisfies ω > (1/2)γ²σ²q² for all |q| ≤ q_max.

    Parameters
    ----------
    q_max: Maximum inventory magnitude.
    gamma: Risk-aversion coefficient.
    sigma: Mid-price volatility.
    """
    return 0.5 * gamma**2 * sigma**2 * (q_max + 1) ** 2


def _check_admissibility(omega: float, gamma: float, sigma: float, q: float | np.ndarray) -> None:
    """Raise if admissibility condition ω > (1/2)γ²σ²q² is violated."""
    threshold = 0.5 * gamma**2 * sigma**2 * np.asarray(q, dtype=float) ** 2
    if np.any(omega <= threshold):
        raise ValueError(
            f"Admissibility violated: ω={omega:.4f} must exceed "
            f"(1/2)γ²σ²q²={np.max(threshold):.4f}"
        )


def infinite_reservation_ask(
    s: float | np.ndarray,
    q: float | int | np.ndarray,
    gamma: float,
    sigma: float,
    omega: float,
) -> float | np.ndarray:
    """
    Stationary reservation ask price for the infinite-horizon agent (Section 2.3).

    r̄^a(s,q) = s + (1/γ) · ln(1 + (1 − 2q)γ²σ² / (2ω − γ²q²σ²))

    Requires ω > (1/2)γ²σ²q².
    """
    _check_admissibility(omega, gamma, sigma, q)
    numerator = (1.0 - 2.0 * q) * gamma**2 * sigma**2
    denominator = 2.0 * omega - gamma**2 * np.asarray(q, dtype=float) ** 2 * sigma**2
    return s + (1.0 / gamma) * np.log(1.0 + numerator / denominator)


def infinite_reservation_bid(
    s: float | np.ndarray,
    q: float | int | np.ndarray,
    gamma: float,
    sigma: float,
    omega: float,
) -> float | np.ndarray:
    """
    Stationary reservation bid price for the infinite-horizon agent (Section 2.3).

    r̄^b(s,q) = s + (1/γ) · ln(1 + (−1 − 2q)γ²σ² / (2ω − γ²q²σ²))

    Requires ω > (1/2)γ²σ²q².
    """
    _check_admissibility(omega, gamma, sigma, q)
    numerator = (-1.0 - 2.0 * q) * gamma**2 * sigma**2
    denominator = 2.0 * omega - gamma**2 * np.asarray(q, dtype=float) ** 2 * sigma**2
    return s + (1.0 / gamma) * np.log(1.0 + numerator / denominator)
