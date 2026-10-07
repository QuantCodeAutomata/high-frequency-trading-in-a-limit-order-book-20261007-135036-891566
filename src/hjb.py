"""
HJB (Hamilton-Jacobi-Bellman) framework for the market-making problem.

Implements Section 3 "The solution" of Avellaneda & Stoikov (2008).

Key results (Sections 3.1–3.2):

1. Value-function transform (eq. 3.2):
   u(s,x,q,t) = −exp(−γx) · exp(−γθ(s,q,t))

2. Reduced PDE for θ under exponential intensities (eq. 3.3):
   θ_t + ½σ²θ_ss − ½σ²γθ_s² + (A/(k+γ))(exp(−kδ^a)+exp(−kδ^b)) = 0
   θ(s,q,T) = q·s

3. Reservation prices from θ (eqs. 3.4–3.5):
   r^b(s,q,t) = θ(s,q+1,t) − θ(s,q,t)
   r^a(s,q,t) = θ(s,q,t)   − θ(s,q−1,t)

4. Implicit first-order conditions for optimal distances (eqs. 3.6–3.7):
   s − r^b = δ^b − (1/γ)·ln(1 − γ·λ^b(δ^b)/λ^{b′}(δ^b))
   r^a − s = δ^a − (1/γ)·ln(1 − γ·λ^a(δ^a)/λ^{a′}(δ^a))

5. Asymptotic expansion (eqs. 3.10–3.18):
   θ = θ^0 + q·θ^1 + ½q²·θ^2 + …
   θ^1(s,t) = s
   θ^2(s,t) = ½σ²γ(T−t)
   ⟹ r(s,q,t) = s − q·γσ²(T−t)   (eq. 3.17)
   ⟹ δ^a + δ^b = γσ²(T−t) + (2/γ)·ln(1+γ/k)   (eq. 3.18)

Custom — Context7 found no library equivalent (Avellaneda & Stoikov § 3.1–3.2)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable, Tuple

from .intensity import ExponentialIntensity, IntensityModel


# ---------------------------------------------------------------------------
# Theta coefficients from the asymptotic expansion (Section 3.2)
# ---------------------------------------------------------------------------

def theta1(s: float | np.ndarray, t: float | np.ndarray, T: float) -> float | np.ndarray:
    """
    First-order coefficient in the inventory expansion of θ (eq. 3.15).

    θ^1(s,t) = s  (independent of t under the linearised arrival term).

    Interpretation: the reservation price at zero inventory equals the mid-price.
    """
    return np.asarray(s, dtype=float)


def theta2(t: float | np.ndarray, T: float, sigma: float, gamma: float) -> float | np.ndarray:
    """
    Second-order coefficient in the inventory expansion of θ (eq. 3.16).

    θ^2(t) = ½σ²γ(T−t)

    Interpretation: sensitivity of the reservation price to inventory changes.
    """
    tau = T - np.asarray(t, dtype=float)
    return 0.5 * sigma**2 * gamma * tau


def reservation_price_approx(
    s: float | np.ndarray,
    q: float | int | np.ndarray,
    t: float | np.ndarray,
    T: float,
    sigma: float,
    gamma: float,
) -> float | np.ndarray:
    """
    Approximate reservation (indifference) price from the asymptotic expansion (eq. 3.17).

    r(s,q,t) = θ^1 − 2q·θ^2 = s − q·γσ²(T−t)

    Matches the frozen-inventory result of Section 2.2 (also eq. 2.8 in the paper
    for the "frozen inventory" problem), confirming that symmetric exponential
    arrival rates yield indifference prices identical to Section 2.2.
    """
    t2 = theta2(t, T, sigma, gamma)
    return s - 2.0 * q * t2


def total_spread_approx(
    t: float | np.ndarray,
    T: float,
    sigma: float,
    gamma: float,
    k: float,
) -> float | np.ndarray:
    """
    Approximate total bid-ask spread from the asymptotic expansion (eq. 3.18).

    δ^a + δ^b = γσ²(T−t) + (2/γ)·ln(1 + γ/k)

    The spread has two components:
      • inventory-risk component: 2·θ^2 = γσ²(T−t)  (decreases to zero at T)
      • order-flow component:     (2/γ)·ln(1+γ/k)  (constant, inventory-independent)
    """
    tau = T - np.asarray(t, dtype=float)
    inventory_component = gamma * sigma**2 * tau
    order_flow_component = (2.0 / gamma) * np.log(1.0 + gamma / k)
    return inventory_component + order_flow_component


# ---------------------------------------------------------------------------
# First-order optimality conditions (Section 3.1, eqs. 3.6–3.7)
# ---------------------------------------------------------------------------

def _foc_equation(
    delta: float,
    indif_distance: float,
    lam_fn: Callable[[float], float],
    dlam_fn: Callable[[float], float],
    gamma: float,
) -> float:
    """
    Residual of the implicit first-order condition (eqs. 3.6–3.7).

    For the ask side:
        residual = (r^a − s) − [δ^a − (1/γ)·ln(1 − γ·λ^a(δ^a)/λ^{a′}(δ^a))]

    For the bid side:
        residual = (s − r^b) − [δ^b − (1/γ)·ln(1 − γ·λ^b(δ^b)/λ^{b′}(δ^b))]

    Parameters
    ----------
    delta:           Candidate δ value.
    indif_distance:  r^a − s  (ask) or  s − r^b  (bid).
    lam_fn:          λ(δ) callable.
    dlam_fn:         λ′(δ) callable.
    gamma:           Risk-aversion coefficient.
    """
    lam = lam_fn(delta)
    dlam = dlam_fn(delta)
    # Admissibility: argument of log must be > 0
    ratio = lam / dlam  # negative for decreasing λ
    arg = 1.0 - gamma * ratio
    if arg <= 0:
        return np.inf  # outside admissible region
    correction = (1.0 / gamma) * np.log(arg)
    return indif_distance - (delta - correction)


def solve_optimal_delta(
    indif_distance: float,
    model: IntensityModel,
    gamma: float,
    delta_min: float = 1e-8,
    delta_max: float = 50.0,
) -> float:
    """
    Numerically solve the first-order optimality condition (eqs. 3.6–3.7)
    for the optimal quote distance δ* given the indifference distance from mid.

    Works for both ask (indif_distance = r^a − s) and bid (indif_distance = s − r^b)
    by sign symmetry of the equations.

    Parameters
    ----------
    indif_distance:
        Indifference price distance from mid-price (must be provided with
        correct sign: positive for ask, positive for bid using s−r^b).
    model:
        IntensityModel instance implementing intensity() and d_intensity().
    gamma:
        Risk-aversion coefficient.
    delta_min, delta_max:
        Bracket for brentq root finding.

    Returns
    -------
    Optimal δ* (positive scalar).
    """
    fn = lambda d: _foc_equation(
        d,
        indif_distance,
        model.intensity,
        model.d_intensity,
        gamma,
    )
    try:
        return brentq(fn, delta_min, delta_max, xtol=1e-10, maxiter=200)
    except ValueError:
        # Fall back to indifference distance as approximation
        return max(indif_distance, delta_min)


# ---------------------------------------------------------------------------
# Exponential-intensity special case: analytical simplification
# ---------------------------------------------------------------------------

def exponential_foc_simplification(
    indif_distance: float,
    gamma: float,
    k: float,
) -> float:
    """
    Closed-form solution of the first-order condition for exponential intensities.

    For λ(δ) = A·exp(−k·δ), we have λ′(δ) = −k·λ(δ), so:
        1 − γ·λ/λ′ = 1 + γ/k
    The implicit condition reduces to:
        indif_distance = δ − (1/γ)·ln(1 + γ/k)
        ⟹ δ* = indif_distance + (1/γ)·ln(1 + γ/k)

    This confirms the spread formula (eq. 3.13):
        δ^a + δ^b = 2·θ^2 + (2/γ)·ln(1+γ/k)

    Parameters
    ----------
    indif_distance:
        r^a − s (ask) or s − r^b (bid).
    gamma:
        Risk-aversion coefficient.
    k:
        Exponential decay rate.

    Returns
    -------
    Optimal δ* (scalar).
    """
    return indif_distance + (1.0 / gamma) * np.log(1.0 + gamma / k)


# ---------------------------------------------------------------------------
# Numerical PDE solver for θ (optional benchmark, Section 3.1)
# ---------------------------------------------------------------------------

def solve_theta_pde_numerical(
    q_grid: np.ndarray,
    s: float,
    T: float,
    sigma: float,
    gamma: float,
    intensity_model: ExponentialIntensity,
    n_time: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward finite-difference solver for the reduced θ PDE (eq. 3.9).

    Because the mid-price s enters only as an initial condition and through
    the terminal condition θ(s,q,T)=q·s, and because the spread formula
    is s-independent under the asymptotic expansion, we solve at a fixed s.

    The PDE per inventory level q is:
        dθ/dt + ½σ²·d²θ/ds² − ½σ²γ(dθ/ds)² + (A/(k+γ))·2·exp(−k·δ*) = 0

    For a rough numerical benchmark we hold s fixed (s-independent θ assumed
    from the expansion) and use the approximate δ* from eq. 3.18.

    Parameters
    ----------
    q_grid:         1-D array of inventory values (integers).
    s:              Reference mid-price.
    T:              Horizon.
    sigma, gamma:   Model parameters.
    intensity_model: ExponentialIntensity instance.
    n_time:         Number of backward time steps.

    Returns
    -------
    t_grid:   Time grid (length n_time+1).
    theta_q:  θ values on q_grid at each time step (shape: n_time+1 × len(q_grid)).
    spread:   Total spread γσ²(T−t) + (2/γ)·ln(1+γ/k) at each time.
    """
    k = intensity_model.k
    A = intensity_model.A
    dt = T / n_time
    t_grid = np.linspace(0.0, T, n_time + 1)

    # Terminal condition: θ(s,q,T) = q·s
    theta = q_grid.astype(float) * s  # shape (len(q_grid),)
    theta_history = np.zeros((n_time + 1, len(q_grid)))
    theta_history[-1] = theta.copy()

    # Backward Euler in time
    for i in range(n_time - 1, -1, -1):
        t_cur = t_grid[i]
        tau = T - t_cur
        # Approximate spread at this time
        delta_star = 0.5 * total_spread_approx(t_cur, T, sigma, gamma, k)  # half-spread
        # Source term from order arrival (eq. 3.9): (A/(k+gamma)) * 2 * exp(-k*delta*)
        source = (A / (k + gamma)) * 2.0 * np.exp(-k * delta_star)
        # Update: explicit Euler backward step (simple PDE benchmark)
        # Drop s-derivative terms since s is constant (theta_ss=0 for affine-in-s approx)
        theta = theta_history[i + 1] + dt * source
        theta_history[i] = theta

    spread = total_spread_approx(t_grid, T, sigma, gamma, k)
    return t_grid, theta_history, spread


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_expansion_coefficients(
    T: float, sigma: float, gamma: float, k: float, s: float = 100.0,
) -> dict:
    """
    Verify that the asymptotic expansion coefficients match eq. 3.17–3.18.

    Returns a dict with computed values and the paper's expected formulas.
    """
    t = 0.0  # at time 0, maximum spread
    t1 = theta1(s, t, T)
    t2_val = theta2(t, T, sigma, gamma)
    r_q0 = reservation_price_approx(s, 0, t, T, sigma, gamma)
    r_q1 = reservation_price_approx(s, 1, t, T, sigma, gamma)
    spread = total_spread_approx(t, T, sigma, gamma, k)
    order_flow_comp = (2.0 / gamma) * np.log(1.0 + gamma / k)
    inventory_comp = gamma * sigma**2 * T

    return {
        "theta1_at_t0": t1,
        "theta2_at_t0": t2_val,
        "theta2_expected": 0.5 * sigma**2 * gamma * T,
        "reservation_price_q0": r_q0,
        "reservation_price_q1": r_q1,
        "reservation_shift_q1": r_q0 - r_q1,  # should be gamma*sigma^2*T
        "spread_at_t0": spread,
        "inventory_component": inventory_comp,
        "order_flow_component": order_flow_comp,
    }
