"""
Experiment 1: Inventory-based quoting strategy — Monte Carlo validation.

Reproduces Section 3.3 "Numerical simulations" of Avellaneda & Stoikov (2008),
Tables 1–3 and Figures 2–4.

Parameters (p. 10):
  s0 = 100, T = 1, σ = 2, dt = 0.005, q0 = 0, k = 1.5, A = 140
  γ ∈ {0.1, 0.01, 0.5},  1 000 Monte Carlo paths per γ

Expected results (Tables 1–3):
  γ = 0.1: Inventory  spread=1.29  profit=62.94  std=5.89   final_q=0.10   std_q=2.80
            Symmetric  spread=1.29  profit=67.21  std=13.43  final_q=−0.018 std_q=8.66
  γ = 0.01: Inventory spread=1.33  profit=66.78  std=8.76   final_q=−0.02  std_q=4.70
             Symmetric spread=1.33  profit=67.36  std=13.40  final_q=−0.31  std_q=8.65
  γ = 0.5: Inventory  spread=1.15  profit=33.92  std=4.72   final_q=−0.02  std_q=1.88
            Symmetric  spread=1.15  profit=66.20  std=14.53  final_q=0.25   std_q=9.06
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SimParams, GAMMA_VALUES
from src.hjb import reservation_price_approx, total_spread_approx
from src.intensity import ExponentialIntensity
from src.simulator import run_comparison, results_table, run_simulation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Paper-reported values for comparison ────────────────────────────────────
PAPER_TABLES = {
    0.1: {
        "inventory": {"spread": 1.29, "profit": 62.94, "std_profit": 5.89,
                      "final_q": 0.10, "std_final_q": 2.80},
        "symmetric": {"spread": 1.29, "profit": 67.21, "std_profit": 13.43,
                      "final_q": -0.018, "std_final_q": 8.66},
    },
    0.01: {
        "inventory": {"spread": 1.33, "profit": 66.78, "std_profit": 8.76,
                      "final_q": -0.02, "std_final_q": 4.70},
        "symmetric": {"spread": 1.33, "profit": 67.36, "std_profit": 13.40,
                      "final_q": -0.31, "std_final_q": 8.65},
    },
    0.5: {
        "inventory": {"spread": 1.15, "profit": 33.92, "std_profit": 4.72,
                      "final_q": -0.02, "std_final_q": 1.88},
        "symmetric": {"spread": 1.15, "profit": 66.20, "std_profit": 14.53,
                      "final_q": 0.25, "std_final_q": 9.06},
    },
}


def run_all_gammas(base_params: SimParams) -> dict:
    """Run simulations for all γ values; return results keyed by γ."""
    results = {}
    for gamma in GAMMA_VALUES:
        params = SimParams(
            s0=base_params.s0,
            T=base_params.T,
            sigma=base_params.sigma,
            dt=base_params.dt,
            q0=base_params.q0,
            gamma=gamma,
            k=base_params.k,
            A=base_params.A,
            n_paths=base_params.n_paths,
            seed=base_params.seed,
        )
        inv, sym = run_comparison(params, gamma=gamma)
        results[gamma] = (inv, sym)
        tbl = results_table(inv, sym)
        table_num = {0.1: 1, 0.01: 2, 0.5: 3}[gamma]
        print(f"\nTable {table_num}: γ = {gamma}")
        print(tbl.to_string(index=False))
    return results


def plot_pnl_histograms(results: dict) -> None:
    """
    Plot P&L histograms for inventory vs symmetric strategy (Figures 2–4).

    Saves one figure per γ to results/.
    """
    fig_num = {0.1: 2, 0.01: 3, 0.5: 4}
    for gamma, (inv, sym) in results.items():
        inv_pnls = [p.pnl for p in inv.paths]
        sym_pnls = [p.pnl for p in sym.paths]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        axes[0].hist(inv_pnls, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        axes[0].set_title(f"Inventory strategy  (γ = {gamma})")
        axes[0].set_xlabel("Terminal P&L")
        axes[0].set_ylabel("Count")

        axes[1].hist(sym_pnls, bins=40, color="coral", edgecolor="white", alpha=0.85)
        axes[1].set_title(f"Symmetric strategy  (γ = {gamma})")
        axes[1].set_xlabel("Terminal P&L")

        fig.suptitle(
            f"Figure {fig_num[gamma]}: P&L distribution, γ = {gamma}  "
            f"[Avellaneda & Stoikov 2008]"
        )
        fig.tight_layout()
        fname = RESULTS_DIR / f"exp1_fig{fig_num[gamma]}_gamma{str(gamma).replace('.','p')}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  Saved {fname.name}")


def plot_sample_path(params: SimParams, gamma: float) -> None:
    """
    Plot one sample path of mid-price, reservation price, bid, and ask (Figure 1).

    Uses the inventory strategy with the given γ.
    """
    rng = np.random.default_rng(params.seed + 99)
    intensity = ExponentialIntensity(A=params.A, k=params.k)
    N = params.N
    dt = params.dt
    sigma = params.sigma
    T = params.T

    S = params.s0
    X = 0.0
    q = float(params.q0)

    ts, mids, reservations, asks, bids = [], [], [], [], []

    for n in range(N):
        t = n * dt
        w = total_spread_approx(t, T, sigma, gamma, params.k)
        r = reservation_price_approx(S, q, t, T, sigma, gamma)
        p_ask = r + w / 2.0
        p_bid = r - w / 2.0

        ts.append(t)
        mids.append(S)
        reservations.append(r)
        asks.append(p_ask)
        bids.append(p_bid)

        delta_a = p_ask - S
        delta_b = S - p_bid
        prob_ask = min(intensity.intensity(delta_a) * dt, 1.0)
        prob_bid = min(intensity.intensity(delta_b) * dt, 1.0)

        if rng.uniform() < prob_ask:
            X += p_ask
            q -= 1
        if rng.uniform() < prob_bid:
            X -= p_bid
            q += 1

        S += rng.choice([-1.0, 1.0]) * sigma * np.sqrt(dt)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, mids, "k-", lw=1.5, label="Mid-price")
    ax.plot(ts, reservations, "b--", lw=1.2, label="Reservation price (indifference)")
    ax.plot(ts, asks, "r-", lw=0.8, alpha=0.7, label="Ask quote")
    ax.plot(ts, bids, "g-", lw=0.8, alpha=0.7, label="Bid quote")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"Figure 1: Sample path  (γ = {gamma})  [Avellaneda & Stoikov 2008]")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fname = RESULTS_DIR / f"exp1_fig1_sample_path_gamma{str(gamma).replace('.','p')}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  Saved {fname.name}")


def compare_with_paper(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame of simulated vs paper-reported results."""
    rows = []
    for gamma, (inv, sym) in results.items():
        table_num = {0.1: 1, 0.01: 2, 0.5: 3}[gamma]
        for strat_name, sim_result in [("inventory", inv), ("symmetric", sym)]:
            paper = PAPER_TABLES[gamma][strat_name]
            rows.append({
                "Table": table_num,
                "γ": gamma,
                "Strategy": strat_name.capitalize(),
                "Paper spread": paper["spread"],
                "Sim spread": round(sim_result.mean_spread, 2),
                "Paper profit": paper["profit"],
                "Sim profit": round(sim_result.mean_pnl, 2),
                "Paper std(profit)": paper["std_profit"],
                "Sim std(profit)": round(sim_result.std_pnl, 2),
                "Paper final_q": paper["final_q"],
                "Sim final_q": round(sim_result.mean_final_q, 3),
                "Paper std(final_q)": paper["std_final_q"],
                "Sim std(final_q)": round(sim_result.std_final_q, 2),
            })
    return pd.DataFrame(rows)


def save_results(results: dict, comparison: pd.DataFrame) -> None:
    """Persist numerical results to results/ as CSV and JSON."""
    comparison.to_csv(RESULTS_DIR / "exp1_comparison.csv", index=False)

    # Aggregate summary tables per gamma
    all_rows = []
    for gamma, (inv, sym) in results.items():
        for strat_name, r in [("Inventory", inv), ("Symmetric", sym)]:
            all_rows.append({
                "gamma": gamma,
                "strategy": strat_name,
                "spread": round(r.mean_spread, 4),
                "profit": round(r.mean_pnl, 4),
                "std_profit": round(r.std_pnl, 4),
                "final_q": round(r.mean_final_q, 4),
                "std_final_q": round(r.std_final_q, 4),
            })
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "exp1_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1: MC simulation comparison")
    parser.add_argument("--n_paths", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_params = SimParams(n_paths=args.n_paths, seed=args.seed)

    print("=" * 60)
    print("Experiment 1: Inventory vs Symmetric strategy (Monte Carlo)")
    print(f"Parameters: s0={base_params.s0}, T={base_params.T}, σ={base_params.sigma}, "
          f"dt={base_params.dt}, k={base_params.k}, A={base_params.A}, "
          f"n_paths={base_params.n_paths}")
    print("=" * 60)

    results = run_all_gammas(base_params)

    print("\n── Sample path plot (Figure 1) ──")
    plot_sample_path(base_params, gamma=0.1)

    print("\n── P&L histograms (Figures 2–4) ──")
    plot_pnl_histograms(results)

    print("\n── Comparison with paper ──")
    comparison = compare_with_paper(results)
    print(comparison.to_string(index=False))

    save_results(results, comparison)
    print("\nDone.")


if __name__ == "__main__":
    main()
