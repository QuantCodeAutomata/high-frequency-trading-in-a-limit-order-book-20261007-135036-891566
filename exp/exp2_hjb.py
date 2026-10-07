"""
Experiment 2: HJB Framework Verification — Avellaneda & Stoikov (2008) §3.1–3.2.

Verifies the asymptotic expansion coefficients, reservation price formula (eq. 3.17),
total spread formula (eq. 3.18), and first-order optimality conditions (eqs. 3.6–3.7).

Expansion convention (§3.2):
    θ(s,q,t) = θ^0 + q·θ^1 − q²·θ^2 + …
    θ^1(s,t) = s                         (eq. 3.15)
    θ^2(t)   = ½σ²γ(T−t)                (eq. 3.16)
    r^a       = θ^1 + (1−2q)·θ^2         (ask reservation price)
    r^b       = θ^1 + (−1−2q)·θ^2        (bid reservation price)
    midpoint  = θ^1 − 2q·θ^2 = s−qγσ²(T−t)  (eq. 3.17)
    δ^a+δ^b   = γσ²(T−t) + (2/γ)ln(1+γ/k)  (eq. 3.18)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hjb import (
    theta1, theta2, reservation_price_approx, total_spread_approx,
    solve_optimal_delta, exponential_foc_simplification,
    verify_expansion_coefficients,
)
from src.intensity import ExponentialIntensity
from src.config import SimParams, DEFAULT_PARAMS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Global parameters ─────────────────────────────────────────────────────────
T, sigma, gamma, k, A, s = 1, 2, 0.1, 1.5, 140, 100
q_range = range(-5, 6)
t_grid = np.linspace(0, 1, 101)

CHECK_TIMES = [0, 0.25, 0.5, 0.75, 1]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── Step 1: verify_expansion_coefficients ────────────────────────────────
    print("=" * 60)
    print("Step 1: verify_expansion_coefficients")
    coef = verify_expansion_coefficients(T=1, sigma=2, gamma=0.1, k=1.5)
    for key, val in coef.items():
        print(f"  {key}: {val}")
    match1 = np.isclose(coef["theta2_at_t0"], coef["theta2_expected"])
    print(f"  theta2_at_t0 == theta2_expected: {match1}")

    # ── Step 2: theta^1 = s (eq 3.15) ────────────────────────────────────────
    print("\nStep 2: theta^1 = s (eq 3.15)")
    for t in CHECK_TIMES:
        val = theta1(s=100, t=t, T=1)
        print(f"  theta1(s=100, t={t}, T=1) = {val}")

    # ── Step 3: theta^2 = 0.5*sigma^2*gamma*(T-t) (eq 3.16) ─────────────────
    print("\nStep 3: theta^2 = 0.5*sigma^2*gamma*(T-t) (eq 3.16)")
    for t in CHECK_TIMES:
        computed = theta2(t=t, T=1, sigma=2, gamma=0.1)
        expected = 0.5 * 4 * 0.1 * (1 - t)
        match = np.isclose(computed, expected)
        print(f"  t={t}: theta2={computed:.6f}, expected={expected:.6f}, match={match}")

    # ── Step 4: spread formula (eq 3.18) ─────────────────────────────────────
    print("\nStep 4: spread formula (eq 3.18)")
    order_flow_comp = (2.0 / gamma) * np.log(1.0 + gamma / k)
    for t in CHECK_TIMES:
        spread = total_spread_approx(t, T=1, sigma=2, gamma=0.1, k=1.5)
        inv_c = gamma * sigma**2 * (T - t)
        print(f"  t={t}: spread={spread:.6f}, inv_component={inv_c:.6f}, "
              f"order_flow_component={order_flow_comp:.6f}")
    spread_at_t0 = total_spread_approx(0, T=1, sigma=2, gamma=0.1, k=1.5)
    formula_t0 = gamma * sigma**2 * T + order_flow_comp
    match4 = np.isclose(spread_at_t0, formula_t0)
    print(f"  spread at t=0 == gamma*sigma^2*T + (2/gamma)*ln(1+gamma/k): {match4}")
    print(f"    computed={spread_at_t0:.6f}, formula={formula_t0:.6f}")

    # ── Step 5: reservation price (eq 3.17) ──────────────────────────────────
    print("\nStep 5: reservation price (eq 3.17)")
    for q in q_range:
        r = reservation_price_approx(100, q, 0, 1, 2, 0.1)
        expected = 100 - q * 0.1 * 4 * 1
        match = np.isclose(r, expected)
        print(f"  q={q:+d}: r={r:.4f}, expected={expected:.4f}, match={match}")

    # ── Step 6: first-order conditions ───────────────────────────────────────
    print("\nStep 6: first-order conditions")
    indif_distance = 0.5
    intensity_model = ExponentialIntensity(140, 1.5)
    delta_numeric = solve_optimal_delta(indif_distance, intensity_model, gamma=0.1)
    delta_analytic = exponential_foc_simplification(indif_distance, gamma=0.1, k=1.5)
    match6 = np.isclose(delta_numeric, delta_analytic, rtol=1e-5)
    print(f"  solve_optimal_delta:            delta*={delta_numeric:.8f}")
    print(f"  exponential_foc_simplification: delta*={delta_analytic:.8f}")
    print(f"  Both give same answer: {match6}")

    # ── Step 7: r^a = theta(q,s,t) - theta(q-1,s,t) ─────────────────────────
    print("\nStep 7: r^a = theta(q,s,t) - theta(q-1,s,t) at t=0")
    t0 = 0.0
    t2_t0 = theta2(t0, T=1, sigma=2, gamma=0.1)
    t1_t0 = theta1(s=100, t=t0, T=1)
    for q in q_range:
        r_a = float(t1_t0) + (1 - 2 * q) * float(t2_t0)
        print(f"  q={q:+d}: r^a={r_a:.4f}")

    # ── Step 8: r^b = theta(q+1,s,t) - theta(q,s,t) ─────────────────────────
    print("\nStep 8: r^b = theta(q+1,s,t) - theta(q,s,t) at t=0")
    for q in q_range:
        r_b = float(t1_t0) + (-1 - 2 * q) * float(t2_t0)
        print(f"  q={q:+d}: r^b={r_b:.4f}")

    # ── Step 9: Plots ─────────────────────────────────────────────────────────
    t_plot = np.linspace(0, 1, 100)

    # (a) theta2 vs time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_plot, theta2(t_plot, T=1, sigma=2, gamma=0.1), color='steelblue', lw=2)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\theta^2(t)$")
    ax.set_title(r"$\theta^2(t) = \frac{1}{2}\sigma^2\gamma(T-t)$  (eq. 3.16)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_theta2_vs_time.png", dpi=120)
    plt.close(fig)
    print("\nSaved exp2_theta2_vs_time.png")

    # (b) spread vs time with labeled components
    spread_vals = total_spread_approx(t_plot, T=1, sigma=2, gamma=0.1, k=1.5)
    inv_comp_vals = gamma * sigma**2 * (T - t_plot)
    of_comp_val = (2.0 / gamma) * np.log(1.0 + gamma / k)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_plot, spread_vals, 'k-', lw=2, label='Total spread')
    ax.plot(t_plot, inv_comp_vals, 'b--', lw=1.5,
            label=r'Inventory-risk: $\gamma\sigma^2(T-t)$')
    ax.axhline(of_comp_val, color='r', ls=':', lw=1.5,
               label=r'Order-flow: $\frac{2}{\gamma}\ln(1+\gamma/k)$')
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"Spread $\delta^a + \delta^b$")
    ax.set_title("Total bid-ask spread decomposition (eq. 3.18)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_spread_vs_time.png", dpi=120)
    plt.close(fig)
    print("Saved exp2_spread_vs_time.png")

    # (c) reservation price vs q for three time slices
    q_arr = np.array(list(q_range))
    fig, ax = plt.subplots(figsize=(8, 4))
    for t_val, color in [(0, 'blue'), (0.5, 'green'), (0.9, 'red')]:
        r_vals = [reservation_price_approx(100, q, t_val, 1, 2, 0.1) for q in q_arr]
        ax.plot(q_arr, r_vals, marker='o', color=color, label=f"t={t_val}")
    ax.axhline(100, color='k', ls='--', lw=0.8, alpha=0.5, label='Mid-price s=100')
    ax.set_xlabel("Inventory $q$")
    ax.set_ylabel("Reservation price $r(s, q, t)$")
    ax.set_title("Reservation price vs inventory (eq. 3.17)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_reservation_vs_q.png", dpi=120)
    plt.close(fig)
    print("Saved exp2_reservation_vs_q.png")

    # ── Step 10: summary CSV ──────────────────────────────────────────────────
    rows = []
    for t_val in t_grid:
        t1_val = float(theta1(s=100, t=t_val, T=1))
        t2_val = float(theta2(t=t_val, T=1, sigma=2, gamma=0.1))
        sp = float(total_spread_approx(t_val, T=1, sigma=2, gamma=0.1, k=1.5))
        for q in q_range:
            r_mid = float(reservation_price_approx(100, q, t_val, 1, 2, 0.1))
            r_ask = t1_val + (1 - 2 * q) * t2_val
            r_bid = t1_val + (-1 - 2 * q) * t2_val
            rows.append({
                "t": round(float(t_val), 6),
                "q": q,
                "theta1": t1_val,
                "theta2": t2_val,
                "r_mid": r_mid,
                "r_ask": r_ask,
                "r_bid": r_bid,
                "spread": sp,
            })
    df = pd.DataFrame(rows, columns=["t", "q", "theta1", "theta2",
                                      "r_mid", "r_ask", "r_bid", "spread"])
    df.to_csv(RESULTS_DIR / "exp2_hjb_results.csv", index=False)
    print(f"Saved exp2_hjb_results.csv ({len(df)} rows)")

    print("\nExp2 complete.")


if __name__ == "__main__":
    main()
