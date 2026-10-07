"""
Monte Carlo simulation of the Avellaneda & Stoikov (2008) market-making strategies.

Implements Section 3.3 "Numerical simulations" (pp. 10–11).

Simulation procedure (as stated in the paper):
  At each time step t:
    1. Compute quotes δ^a and δ^b from the state variables.
    2. At time t+dt:
       • With probability λ^a(δ^a)·dt the agent sells one share:
           inventory q decreases by 1, wealth increases by s + δ^a (= p^a).
       • With probability λ^b(δ^b)·dt the agent buys one share:
           inventory q increases by 1, wealth decreases by s − δ^b (= p^b).
       • Mid-price is updated by a random increment ±σ√dt (Bernoulli ±1).
    3. Terminal P&L: Π = X_T + q_T · S_T.

Two strategies are compared:
  • Inventory strategy: centers quotes at the reservation price r = S − q·γσ²(T−t).
  • Symmetric benchmark: centers the same spread at the mid-price S.

Custom — Context7 found no library equivalent (Avellaneda & Stoikov § 3.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import SimParams
from .hjb import reservation_price_approx, total_spread_approx
from .intensity import ExponentialIntensity


@dataclass
class PathResult:
    """Results for a single Monte Carlo path."""
    pnl: float          # Terminal P&L = X_T + q_T * S_T
    final_q: float      # Terminal inventory q_T
    spread: float       # Average total spread used over the path
    n_ask_exec: int     # Number of ask executions
    n_bid_exec: int     # Number of bid executions
    double_exec: int    # Number of time steps with simultaneous bid+ask execution


@dataclass
class SimulationResult:
    """Aggregate statistics across all Monte Carlo paths."""
    strategy: str
    gamma: float
    mean_spread: float
    mean_pnl: float
    std_pnl: float
    mean_final_q: float
    std_final_q: float
    paths: List[PathResult]

    def to_series(self) -> pd.Series:
        return pd.Series({
            "Strategy": self.strategy,
            "Spread": round(self.mean_spread, 2),
            "Profit": round(self.mean_pnl, 2),
            "std(Profit)": round(self.std_pnl, 2),
            "Final q": round(self.mean_final_q, 3),
            "std(Final q)": round(self.std_final_q, 2),
        })


def _simulate_one_path(
    params: SimParams,
    strategy: str,
    rng: np.random.Generator,
    gamma: float,
) -> PathResult:
    """
    Simulate one Monte Carlo path for the specified strategy.

    Parameters
    ----------
    params:   Simulation parameters.
    strategy: 'inventory' or 'symmetric'.
    rng:      Random generator for reproducibility.
    gamma:    Risk-aversion coefficient (overrides params.gamma).

    Returns
    -------
    PathResult for this path.
    """
    assert strategy in ("inventory", "symmetric"), f"Unknown strategy: {strategy}"

    S = params.s0          # current mid-price
    X = 0.0               # cash wealth (initial wealth irrelevant for P&L differences)
    q = float(params.q0)  # inventory
    intensity = ExponentialIntensity(A=params.A, k=params.k)
    N = params.N
    dt = params.dt
    sigma = params.sigma
    T = params.T

    spread_sum = 0.0
    n_ask = 0
    n_bid = 0
    double_exec = 0

    # Pre-draw random increments for mid-price (Bernoulli ±1)
    xi = rng.choice([-1.0, 1.0], size=N)  # ±σ√dt increments
    # Pre-draw uniform samples for Bernoulli execution tests
    u_ask = rng.uniform(0, 1, size=N)
    u_bid = rng.uniform(0, 1, size=N)

    for n in range(N):
        t = n * dt
        tau = T - t

        # ── 1. Compute quotes ──
        # Approximate total spread (eq. 3.18)
        w = total_spread_approx(t, T, sigma, gamma, params.k)

        if strategy == "inventory":
            # Reservation price (eq. 3.17): r = S − q·γσ²(T−t)
            r = reservation_price_approx(S, q, t, T, sigma, gamma)
            p_ask = r + w / 2.0   # eq. 3.18 quoted around r
            p_bid = r - w / 2.0
        else:  # symmetric
            # Same spread but centred at mid-price (Section 3.3 benchmark)
            p_ask = S + w / 2.0
            p_bid = S - w / 2.0

        delta_a = p_ask - S   # distance from mid to ask
        delta_b = S - p_bid   # distance from mid to bid

        # ── 2. Execution probabilities ──
        lambda_a = intensity.intensity(delta_a)
        lambda_b = intensity.intensity(delta_b)
        prob_ask = lambda_a * dt
        prob_bid = lambda_b * dt

        # Clip to [0,1] for Bernoulli validity (λdt > 1 is extremely rare
        # with paper parameters; clip and note diagnostically)
        prob_ask = min(prob_ask, 1.0)
        prob_bid = min(prob_bid, 1.0)

        # ── 3. Execute limit orders (independent Bernoulli draws) ──
        ask_exec = u_ask[n] < prob_ask
        bid_exec = u_bid[n] < prob_bid

        if ask_exec and bid_exec:
            double_exec += 1

        if ask_exec:
            X += p_ask   # sell one share at p_ask = S + δ^a (paper: "wealth increases by s+δ^a")
            q -= 1
            n_ask += 1

        if bid_exec:
            X -= p_bid   # buy one share at p_bid = S − δ^b (paper: "wealth decreases by s−δ^b")
            q += 1
            n_bid += 1

        # ── 4. Update mid-price ──
        S += xi[n] * sigma * np.sqrt(dt)

        spread_sum += w

    # Terminal mark-to-market P&L
    pnl = X + q * S

    return PathResult(
        pnl=pnl,
        final_q=q,
        spread=spread_sum / N,
        n_ask_exec=n_ask,
        n_bid_exec=n_bid,
        double_exec=double_exec,
    )


def run_simulation(
    params: SimParams,
    strategy: str,
    gamma: Optional[float] = None,
) -> SimulationResult:
    """
    Run the Monte Carlo simulation for a given strategy and risk-aversion γ.

    Parameters
    ----------
    params:   Simulation parameters (from SimParams).
    strategy: 'inventory' or 'symmetric'.
    gamma:    Override risk-aversion coefficient (uses params.gamma if None).

    Returns
    -------
    SimulationResult with aggregate statistics across all paths.

    Notes
    -----
    mean_spread stores the order-flow (inventory-independent) spread component
    (2/γ)·ln(1 + γ/k), which is what Tables 1–3 of the paper report (p. 11).
    The total spread δ^a + δ^b = γσ²(T−t) + order_flow_spread varies in time;
    the order-flow component equals the spread at T, i.e. the minimum spread
    of the strategy.
    """
    if gamma is None:
        gamma = params.gamma

    # Order-flow spread component (2/γ)·ln(1+γ/k): the inventory-independent
    # term that the paper uses as the "spread" in Tables 1–3 (p. 11).
    order_flow_spread = (2.0 / gamma) * np.log(1.0 + gamma / params.k)

    rng = np.random.default_rng(params.seed)
    paths: List[PathResult] = [
        _simulate_one_path(params, strategy, rng, gamma)
        for _ in range(params.n_paths)
    ]

    pnls = np.array([p.pnl for p in paths])
    final_qs = np.array([p.final_q for p in paths])

    return SimulationResult(
        strategy=strategy,
        gamma=gamma,
        mean_spread=order_flow_spread,
        mean_pnl=float(np.mean(pnls)),
        std_pnl=float(np.std(pnls, ddof=1)),
        mean_final_q=float(np.mean(final_qs)),
        std_final_q=float(np.std(final_qs, ddof=1)),
        paths=paths,
    )


def run_comparison(
    params: SimParams,
    gamma: float,
) -> Tuple[SimulationResult, SimulationResult]:
    """
    Run both inventory and symmetric strategies with the same parameters and seed.

    Parameters
    ----------
    params: Simulation parameters.
    gamma:  Risk-aversion coefficient (overrides params.gamma).

    Returns
    -------
    (inventory_result, symmetric_result)
    """
    inv = run_simulation(params, "inventory", gamma=gamma)
    sym = run_simulation(params, "symmetric", gamma=gamma)
    return inv, sym


def results_table(
    inv: SimulationResult,
    sym: SimulationResult,
) -> pd.DataFrame:
    """
    Format results as the paper's Table 1/2/3 layout (Section 3.3, p. 11–12).

    Columns: Strategy, Spread, Profit, std(Profit), Final q, std(Final q).
    """
    rows = [inv.to_series(), sym.to_series()]
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df
