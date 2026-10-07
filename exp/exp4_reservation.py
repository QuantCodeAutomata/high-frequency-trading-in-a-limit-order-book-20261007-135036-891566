"""
Experiment 4: Reservation (indifference) prices — Avellaneda & Stoikov (2008).

Covers Sections 2.2 (finite-horizon, eqs. 2.3–2.7) and 2.3 (infinite-horizon
stationary) of the paper.

Parameters: s=100, T=1, σ=2, γ=0.1, q_max=5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reservation import (
    finite_value,
    finite_reservation_ask,
    finite_reservation_bid,
    finite_reservation_mid,
    omega_bounded,
    infinite_reservation_ask,
    infinite_reservation_bid,
)
from src.config import SimParams, DEFAULT_PARAMS, INF_HORIZON_QMAX

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed experiment parameters
S = 100.0
T = 1.0
SIGMA = 2.0
GAMMA = 0.1
Q_VALS = list(range(-5, 6))
T_VALS = [0.0, 0.25, 0.5, 0.75, 0.9]
Q_MAX = 5


def main():
    print("=" * 72)
    print("Experiment 4: Reservation prices (Avellaneda & Stoikov 2008, §2.2–2.3)")
    print(f"Parameters: s={S}, T={T}, σ={SIGMA}, γ={GAMMA}, q_max={Q_MAX}")
    print("=" * 72)

    # ─────────────────────────────────────────────────────────────────────────
    # PART A — Finite-horizon (Section 2.2, eqs. 2.3–2.7)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── PART A: Finite-horizon reservation prices (Section 2.2) ──\n")

    # ── Step 1: Table + formula verification ─────────────────────────────────
    print("Step 1: Reservation price table and formula check")
    print(f"  r_mid formula: s − q·γσ²(T−t) = 100 − q·0.1·4·(1−t)\n")
    print(f"  {'q':>4}  {'t':>6}  {'r_ask':>10}  {'r_bid':>10}  {'r_mid':>10}  {'formula':>10}")
    print("  " + "-" * 58)

    table_rows = []
    max_abs_err = 0.0

    for q in [-3, -2, -1, 0, 1, 2, 3]:
        for t in [0.0, 0.25, 0.5, 0.75]:
            r_ask = finite_reservation_ask(S, q, t, T, SIGMA, GAMMA)
            r_bid = finite_reservation_bid(S, q, t, T, SIGMA, GAMMA)
            r_mid = finite_reservation_mid(S, q, t, T, SIGMA, GAMMA)
            r_mid_formula = S - q * GAMMA * SIGMA**2 * (T - t)
            err = abs(r_mid - r_mid_formula)
            max_abs_err = max(max_abs_err, err)
            print(f"  {q:>4}  {t:>6.2f}  {r_ask:>10.4f}  {r_bid:>10.4f}  "
                  f"{r_mid:>10.4f}  {r_mid_formula:>10.4f}")
            table_rows.append({
                "q": q,
                "t": t,
                "r_ask": r_ask,
                "r_bid": r_bid,
                "r_mid": r_mid,
                "r_mid_formula_check": r_mid_formula,
            })

    print(f"\n  Max absolute error |r_mid − formula|: {max_abs_err:.2e}")

    # ── Step 2: Verify Definition 1 numerically (q=2, t=0.5) ─────────────────
    print("\nStep 2: Verify Definition 1 — bid indifference (q=2, t=0.5)")
    q2, t2 = 2, 0.5
    r_b = finite_reservation_bid(S, q2, t2, T, SIGMA, GAMMA)
    val_buy    = finite_value(-r_b, S, q2 + 1, t2, T, SIGMA, GAMMA)
    val_notrade = finite_value(0.0,  S, q2,     t2, T, SIGMA, GAMMA)
    rel_err = abs(val_buy - val_notrade) / abs(val_notrade)
    print(f"  r_bid(q=2, t=0.5)               = {r_b:.6f}")
    print(f"  v(−r_b, s, q+1=3, t=0.5) = {val_buy:.6e}")
    print(f"  v(  0,  s, q=2,   t=0.5) = {val_notrade:.6e}")
    print(f"  Relative error              = {rel_err:.2e}")

    # ── Step 3: Sign verification ─────────────────────────────────────────────
    print("\nStep 3: Sign of r_mid relative to s  (t=0.5)")
    print(f"  {'q':>4}  {'r_mid':>10}  {'r_mid − s':>12}  Direction")
    for q in [-3, -1, 0, 1, 3]:
        r_mid = finite_reservation_mid(S, q, 0.5, T, SIGMA, GAMMA)
        diff = r_mid - S
        if q > 0:
            direction = "< 0  ✓ (pos. inv. → r_mid < s)"
        elif q < 0:
            direction = "> 0  ✓ (neg. inv. → r_mid > s)"
        else:
            direction = "= 0  ✓ (zero inv. → r_mid = s)"
        print(f"  {q:>4}  {r_mid:>10.4f}  {diff:>12.4f}  {direction}")

    # ── Step 4: Plot (r_mid − s) vs q for t = 0.0, 0.5, 0.9 ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in [0.0, 0.5, 0.9]:
        r_mid_arr = np.array([
            finite_reservation_mid(S, q, t, T, SIGMA, GAMMA) for q in Q_VALS
        ])
        ax.plot(Q_VALS, r_mid_arr - S, marker="o", ms=5, label=f"t = {t}")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel(r"$r(s,q,t) - s$")
    ax.set_title(
        "Finite-horizon: reservation mid-price adjustment vs inventory\n"
        r"$r(s,q,t) - s = -q\,\gamma\sigma^2(T-t)$  [A&S 2008, §2.2]"
    )
    ax.legend(title="Time t")
    fig.tight_layout()
    fname4 = RESULTS_DIR / "exp4_finite_reservation_vs_q.png"
    fig.savefig(fname4, dpi=120)
    plt.close(fig)
    print(f"\nStep 4: Saved {fname4.name}")

    # ── Step 5: Plot (r_ask − r_bid) vs t for q=0 ────────────────────────────
    t_arr = np.linspace(0.0, 1.0, 300)
    spread_arr = np.array([
        finite_reservation_ask(S, 0, t, T, SIGMA, GAMMA)
        - finite_reservation_bid(S, 0, t, T, SIGMA, GAMMA)
        for t in t_arr
    ])
    theory_arr = GAMMA * SIGMA**2 * (T - t_arr)   # = 0.4*(1−t)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_arr, spread_arr, lw=2.5, label=r"$r^a - r^b$  (q = 0)")
    ax.plot(t_arr, theory_arr, "r--", lw=1.8,
            label=r"$\gamma\sigma^2(T-t) = 0.4(1-t)$")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Reservation spread")
    ax.set_title("Finite-horizon reservation spread vs time (q=0)  [A&S 2008, §2.2]")
    ax.legend()
    fig.tight_layout()
    fname5 = RESULTS_DIR / "exp4_finite_spread_vs_t.png"
    fig.savefig(fname5, dpi=120)
    plt.close(fig)
    print(f"Step 5: Saved {fname5.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART B — Infinite-horizon stationary (Section 2.3)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── PART B: Infinite-horizon reservation prices (Section 2.3) ──\n")

    # ── Step 6: omega ─────────────────────────────────────────────────────────
    omega = omega_bounded(q_max=Q_MAX, gamma=GAMMA, sigma=SIGMA)
    print(f"Step 6: ω = 0.5·γ²·σ²·(q_max+1)²")
    print(f"  omega_bounded(q_max={Q_MAX}, γ={GAMMA}, σ={SIGMA}) = {omega:.6f}")

    # ── Step 7: Admissibility ─────────────────────────────────────────────────
    print(f"\nStep 7: Admissibility — ω={omega:.4f} vs 0.5·γ²·σ²·q² for all q")
    print(f"  {'q':>4}  {'0.5·γ²·σ²·q²':>14}  {'ω':>10}  {'ω > threshold':>14}")
    print("  " + "-" * 48)
    all_ok = True
    for q in range(-Q_MAX, Q_MAX + 1):
        threshold = 0.5 * GAMMA**2 * SIGMA**2 * q**2
        ok = omega > threshold
        if not ok:
            all_ok = False
        print(f"  {q:>4}  {threshold:>14.4f}  {omega:>10.4f}  {'Yes' if ok else 'FAIL':>14}")
    print(f"\n  All admissible: {all_ok}")

    # ── Steps 8 & 9: Infinite vs finite table ─────────────────────────────────
    print("\nSteps 8–9: Reservation price table — infinite-horizon vs finite (t=0)")
    hdr = f"  {'q':>4}  {'r_ask_inf':>12}  {'r_bid_inf':>12}  {'r_mid_inf':>12}  {'r_mid_fin(t=0)':>16}"
    print(hdr)
    print("  " + "-" * 64)
    for q in range(-Q_MAX, Q_MAX + 1):
        r_ask_inf = infinite_reservation_ask(S, q, GAMMA, SIGMA, omega)
        r_bid_inf = infinite_reservation_bid(S, q, GAMMA, SIGMA, omega)
        r_mid_inf = (r_ask_inf + r_bid_inf) / 2.0
        r_mid_fin = finite_reservation_mid(S, q, 0.0, T, SIGMA, GAMMA)
        print(f"  {q:>4}  {r_ask_inf:>12.4f}  {r_bid_inf:>12.4f}  "
              f"{r_mid_inf:>12.4f}  {r_mid_fin:>16.4f}")

    # ── Step 10: Plot infinite vs finite r_mid ────────────────────────────────
    q_range = list(range(-Q_MAX, Q_MAX + 1))
    r_mid_inf_arr = np.array([
        (infinite_reservation_ask(S, q, GAMMA, SIGMA, omega)
         + infinite_reservation_bid(S, q, GAMMA, SIGMA, omega)) / 2.0
        for q in q_range
    ])
    r_mid_fin_arr = np.array([
        finite_reservation_mid(S, q, 0.0, T, SIGMA, GAMMA)
        for q in q_range
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q_range, r_mid_inf_arr - S, "b-o", ms=6,
            label="Infinite-horizon $\\bar{r}_{mid}$")
    ax.plot(q_range, r_mid_fin_arr - S, "r--s", ms=6,
            label="Finite-horizon $r_{mid}$ (t=0)")
    ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel(r"$r_{mid} - s$")
    ax.set_title("Infinite- vs finite-horizon reservation mid-price  [A&S 2008, §2.3]")
    ax.legend()
    fig.tight_layout()
    fname10 = RESULTS_DIR / "exp4_infinite_vs_finite_reservation.png"
    fig.savefig(fname10, dpi=120)
    plt.close(fig)
    print(f"\nStep 10: Saved {fname10.name}")

    # ── Step 11: Sensitivity to γ (q=1, t=0) ─────────────────────────────────
    print("\nStep 11: Sensitivity to γ — finite r_mid(s=100, q=1, t=0, T=1, σ=2, γ) − 100")
    print(f"  {'γ':>8}  {'r_mid − s':>12}  {'−γσ²T (formula)':>18}")
    print("  " + "-" * 44)
    for g in [0.01, 0.1, 0.5, 1.0]:
        adj = finite_reservation_mid(S, 1, 0.0, T, SIGMA, g) - S
        formula = -g * SIGMA**2 * T           # = −4γ  (T=1, σ²=4)
        print(f"  {g:>8.2f}  {adj:>12.6f}  {formula:>18.6f}")

    # ── Step 12: Save CSV ─────────────────────────────────────────────────────
    df = pd.DataFrame(table_rows)
    csv_path = RESULTS_DIR / "exp4_reservation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nStep 12: Results saved to {csv_path}")

    # ── Step 13 ───────────────────────────────────────────────────────────────
    print("\nExp4 complete.")


if __name__ == "__main__":
    main()
