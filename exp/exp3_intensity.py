"""
Avellaneda & Stoikov (2008) – Section 2.5 Intensity Derivation
Reproduces the analytical derivation of eq. 2.11 and compares the
exponential and power-law execution-intensity models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intensity import ExponentialIntensity, PowerLawIntensity, compare_intensities
from src.config import SimParams, DEFAULT_PARAMS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # ------------------------------------------------------------------
    # 1. Analytical derivation of eq. 2.11 (logarithmic impact case)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Section 2.5 – Derivation of λ(δ) = A·exp(−k·δ)  (eq. 2.11)")
    print("=" * 60)
    print("Step 1: lambda(delta) = Lambda * P(Delta_p > delta)")
    print("Step 2: f^Q(x) ~ x^{-1-alpha}, so P(Q > x) = x^{-alpha} / alpha (up to constant)")
    print("Step 3: Delta_p = (1/K)*ln(Q) => Q = exp(K*delta), so P(Delta_p > delta) = P(Q > exp(K*delta))")
    print("Step 4: P(Q > exp(K*delta)) = integral_{exp(K*delta)}^inf x^{-1-alpha} dx = exp(-K*alpha*delta)/alpha")
    print("Step 5: lambda(delta) = (Lambda/alpha)*exp(-alpha*K*delta) = A*exp(-k*delta), A=Lambda/alpha, k=alpha*K")
    print()

    # ------------------------------------------------------------------
    # 2. Calibration verification
    # ------------------------------------------------------------------
    Lambda, alpha, K = 210.0, 1.5, 1.0
    A_expected = Lambda / alpha          # 140.0
    k_expected = alpha * K               # 1.5

    calibrated = ExponentialIntensity.from_micro(Lambda=Lambda, alpha=alpha, K=K)

    print("Calibration: Lambda=210.0, alpha=1.5, K=1.0")
    print(f"  Expected  A = Lambda/alpha = {A_expected:.4f}")
    print(f"  Expected  k = alpha*K      = {k_expected:.4f}")
    print(f"  from_micro A = {calibrated.A:.4f}  (match: {np.isclose(calibrated.A, A_expected)})")
    print(f"  from_micro k = {calibrated.k:.4f}  (match: {np.isclose(calibrated.k, k_expected)})")
    print()

    # ------------------------------------------------------------------
    # 3. Construct models
    # ------------------------------------------------------------------
    exp_model = ExponentialIntensity(A=140.0, k=1.5)
    pl_model  = PowerLawIntensity(B=50.0, alpha=1.5, beta=0.5)

    # ------------------------------------------------------------------
    # 4. Evaluate on δ grid
    # ------------------------------------------------------------------
    delta_grid = np.linspace(0.01, 5.0, 500)
    results = compare_intensities(delta_grid, exp_model, pl_model)

    # ------------------------------------------------------------------
    # 5. Summary table at selected δ values
    # ------------------------------------------------------------------
    dt = DEFAULT_PARAMS.dt   # 0.005
    print(f"Intensity table  (dt = {dt})")
    print(f"{'delta':>6}  {'lambda_exp':>12}  {'lambda_pl':>12}  {'prob_exp':>10}  {'prob_pl':>10}")
    print("-" * 60)
    all_probs_lt1 = True
    for d in [0.1, 0.5, 1.0, 1.5, 2.0]:
        lam_exp = exp_model.intensity(d)
        lam_pl  = pl_model.intensity(d)
        prob_exp = lam_exp * dt
        prob_pl  = lam_pl  * dt
        if prob_exp >= 1.0 or prob_pl >= 1.0:
            all_probs_lt1 = False
        print(f"{d:>6.1f}  {lam_exp:>12.6f}  {lam_pl:>12.6f}  {prob_exp:>10.6f}  {prob_pl:>10.6f}")
    print(f"\nAll execution probabilities (λ·dt) < 1: {all_probs_lt1}")
    print()

    # ------------------------------------------------------------------
    # 6. Plot: intensity comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["delta"], results["lambda_exp"], label=r"Exponential: $A e^{-k\delta}$", linewidth=2)
    ax.plot(results["delta"], results["lambda_pl"],  label=r"Power-law: $B\,\delta^{-\alpha/\beta}$", linewidth=2, linestyle="--")
    ax.set_xlabel(r"Quote distance $\delta$", fontsize=12)
    ax.set_ylabel(r"Execution intensity $\lambda(\delta)$", fontsize=12)
    ax.set_title("A&S (2008) Section 2.5 – Intensity Model Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(delta_grid[0], delta_grid[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "exp3_intensity_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: results/exp3_intensity_comparison.png")

    # ------------------------------------------------------------------
    # 7. Plot: execution probability λ(δ)·dt vs δ
    # ------------------------------------------------------------------
    exec_prob_exp = results["lambda_exp"] * dt
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(results["delta"], exec_prob_exp, linewidth=2,
             label=r"$\lambda_{\rm exp}(\delta)\cdot dt$")
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Prob = 1 (upper bound)")
    ax2.set_xlabel(r"Quote distance $\delta$", fontsize=12)
    ax2.set_ylabel(r"Execution probability $\lambda(\delta)\cdot dt$", fontsize=12)
    ax2.set_title(rf"A&S (2008) – Execution Probability per Step ($dt = {dt}$)", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_xlim(delta_grid[0], delta_grid[-1])
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "exp3_exec_prob.png", dpi=150)
    plt.close(fig2)
    print("Saved: results/exp3_exec_prob.png")
    print()

    # ------------------------------------------------------------------
    # 8. λ′/λ ratio
    # ------------------------------------------------------------------
    delta_ref = 1.0
    ratio_exp = exp_model.d_intensity(delta_ref) / exp_model.intensity(delta_ref)
    ratio_pl  = pl_model.d_intensity(delta_ref)  / pl_model.intensity(delta_ref)
    print("λ′/λ ratio (logarithmic sensitivity of intensity):")
    print(f"  Exponential model at δ={delta_ref}: λ′/λ = {ratio_exp:.4f}  (constant = -k = {-exp_model.k:.4f})")
    print(f"  Power-law  model at δ={delta_ref}: λ′/λ = {ratio_pl:.4f}  (theoretical = -(α/β)/δ = {-(pl_model.alpha/pl_model.beta)/delta_ref:.4f})")
    print()

    # ------------------------------------------------------------------
    # 9. Order-flow spread component (2/γ)·ln(1 + γ/k) as k varies
    # ------------------------------------------------------------------
    gamma = DEFAULT_PARAMS.gamma   # 0.1
    print(f"Order-flow spread component (2/γ)·ln(1 + γ/k)  [γ = {gamma}]")
    print(f"{'k':>6}  {'spread_component':>18}")
    print("-" * 28)
    spread_rows = []
    for k_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
        component = (2.0 / gamma) * np.log(1.0 + gamma / k_val)
        spread_rows.append({"k": k_val, "spread_component": component})
        print(f"{k_val:>6.1f}  {component:>18.6f}")
    print()

    # ------------------------------------------------------------------
    # 10. Save CSV results
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "delta":        results["delta"],
        "lambda_exp":   results["lambda_exp"],
        "lambda_pl":    results["lambda_pl"],
        "exec_prob_exp": results["lambda_exp"] * dt,
        "exec_prob_pl":  results["lambda_pl"]  * dt,
    })
    csv_path = out_dir / "exp3_intensity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: results/exp3_intensity_results.csv  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # 11. Done
    # ------------------------------------------------------------------
    print()
    print("Exp3 complete.")


if __name__ == "__main__":
    main()
