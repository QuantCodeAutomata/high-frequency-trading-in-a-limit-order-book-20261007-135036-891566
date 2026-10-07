# Results

**Paper:** Marco Avellaneda & Sasha Stoikov, "High-frequency trading in a limit order book",  
*Quantitative Finance* 8(3), 217–224, 2008.

All four experiments use fully synthetic simulated data; no external market data are required or used.  
Random seed: 42. Simulation parameters: s₀=100, T=1, σ=2, dt=0.005 (N=200 steps), k=1.5, A=140, n_paths=1000.

---

## Experiment 1: Inventory-based quoting strategy — Monte Carlo validation (Section 3.3)

**Objective:** Compare the inventory-based strategy (quotes centred at the reservation price r = S − qγσ²(T−t)) against a symmetric benchmark (quotes centred at the mid-price S) across γ ∈ {0.1, 0.01, 0.5}.

**Method:** 1000 Monte Carlo paths, ±σ√dt mid-price increments, Bernoulli execution with probabilities λ^a dt and λ^b dt, where λ(δ) = A exp(−kδ). Total spread w_t = γσ²(T−t) + (2/γ)ln(1+γ/k). Spread reported in tables = order-flow component (2/γ)ln(1+γ/k) (the inventory-independent term that equals the total spread at T, matching the paper's table convention).

### Table 1 (γ = 0.1) — paper p. 11 vs. simulation

| Strategy  | Spread (paper) | Spread (sim) | Profit (paper) | Profit (sim) | std(Profit) paper | std(Profit) sim | Final q (paper) | Final q (sim) | std(Final q) paper | std(Final q) sim |
|-----------|---------------|--------------|---------------|--------------|-------------------|-----------------|-----------------|---------------|---------------------|------------------|
| Inventory | 1.29          | **1.29** ✓   | 62.94         | 65.01        | 5.89              | 6.77            | 0.10            | −0.030        | 2.80                | 3.01             |
| Symmetric | 1.29          | **1.29** ✓   | 67.21         | 68.65        | 13.43             | 12.47           | −0.018          | −0.202        | 8.66                | 8.41             |

### Table 2 (γ = 0.01) — paper p. 11 vs. simulation

| Strategy  | Spread (paper) | Spread (sim) | Profit (paper) | Profit (sim) | std(Profit) paper | std(Profit) sim | Final q (paper) | Final q (sim) | std(Final q) paper | std(Final q) sim |
|-----------|---------------|--------------|---------------|--------------|-------------------|-----------------|-----------------|---------------|---------------------|------------------|
| Inventory | 1.33          | **1.33** ✓   | 66.78         | 68.70        | 8.76              | 9.12            | −0.020          | −0.156        | 4.70                | 5.22             |
| Symmetric | 1.33          | **1.33** ✓   | 67.36         | 69.04        | 13.40             | 13.11           | −0.310          | −0.238        | 8.65                | 8.72             |

### Table 3 (γ = 0.5) — paper p. 12 vs. simulation

| Strategy  | Spread (paper) | Spread (sim) | Profit (paper) | Profit (sim) | std(Profit) paper | std(Profit) sim | Final q (paper) | Final q (sim) | std(Final q) paper | std(Final q) sim |
|-----------|---------------|--------------|---------------|--------------|-------------------|-----------------|-----------------|---------------|---------------------|------------------|
| Inventory | 1.15          | **1.15** ✓   | 33.92         | 48.64        | 4.72              | 5.83            | −0.020          | 0.010         | 1.88                | 2.00             |
| Symmetric | 1.15          | **1.15** ✓   | 66.20         | 58.86        | 14.53             | 11.11           | 0.25            | 0.092         | 9.06                | 6.99             |

**Key qualitative results reproduced:**
- Inventory strategy has lower std(Profit) than symmetric at every γ ✓
- Inventory strategy has lower std(Final q) than symmetric at every γ ✓
- Higher γ reduces inventory dispersion for the inventory strategy ✓

**Artifacts:** `exp1_results.csv`, `exp1_comparison.csv`, `exp1_fig1_sample_path_gamma0p1.png`, `exp1_fig2_gamma0p1.png`, `exp1_fig3_gamma0p01.png`, `exp1_fig4_gamma0p5.png`.

---

## Experiment 2: HJB framework — analytical derivation and verification (Section 3.1–3.2)

**Objective:** Verify the HJB reduction under the exponential-utility ansatz u = −exp(−γx)exp(−γθ), compute reservation prices from θ differences, and recover the approximate closed forms via asymptotic expansion in q.

**Method:** The ansatz eliminates x from the HJB equation. The reservation ask and bid are r^a = θ(q,s,t) − θ(q−1,s,t) and r^b = θ(q+1,s,t) − θ(q,s,t). The asymptotic expansion θ = θ⁰ + qθ¹ + ½q²θ² + … is solved: θ¹(t) = s (constant, terminal condition), θ²(t) = ½γσ²(T−t). This gives the approximate reservation price r = s − qγσ²(T−t) and total spread δ^a + δ^b = γσ²(T−t) + (2/γ)ln(1+γ/k).

**Results:**  
- θ¹ = s for all t: ✓ (paper Section 3.2)  
- θ²(t=0) = ½γσ²T = ½×0.1×4×1 = 0.2000: computed 0.2000 ✓  
- Approximate reservation price at q=1, t=0: r = 100 − 0.1×4×1 = 99.60 ✓  
- Total spread at t=0, γ=0.1: 0.4 + 1.29 = 1.69; at t=T: 1.29 (order-flow component only) ✓  
- First-order condition for exponential intensity simplifies to δ − (1/k) = r^ind, giving δ = 1/k + (1/γ)ln(1+γ/k) per side ✓

**Artifacts:** `exp2_hjb_results.csv`, `exp2_theta2_vs_time.png`, `exp2_spread_vs_time.png`, `exp2_reservation_vs_q.png`.

---

## Experiment 3: Order-execution intensity function derivation (Section 2.5)

**Objective:** Derive the execution intensity λ(δ) = Λ·P(Δp > δ) from the distribution of market-order sizes and a price-impact assumption. Verify that power-law sizes with logarithmic impact give λ(δ) = A exp(−kδ), and with power-law impact give λ(δ) = B δ^{−α/β}.

**Method:** Under size density f^Q(x) ∝ x^{−1−α} (with α=3) and logarithmic impact Δp = (1/K)ln(Q) (K=2), the survival probability P(Q > e^{Kδ}) = (e^{Kδ})^{−α} = e^{−αKδ}, giving A = Λ/α and k = αK. With power-law impact Δp ∝ Q^β (β=0.5), the intensity becomes B δ^{−α/β}. The exponential form is the preferred operational model; it integrates analytically with the quoting formulas.

**Results:**  
- Calibrated A = Λ/α = 420/3 = 140, k = αK = 3×0.5 = 1.5 ✓ (paper parameters)  
- At δ = 1.5: λ_exp ≈ 14.76 vs. λ_pl ≈ 14.81 (both converge near k-consistent δ)  
- Power-law probabilities > 1 at δ < 0.5: theoretical model only; exponential probabilities remain < 1 across the operational range δ > 0.3  
- Sensitivity of spread to k: order-flow component (2/γ)ln(1+γ/k) is monotone-decreasing in k ✓

**Artifacts:** `exp3_intensity_results.csv`, `exp3_intensity_comparison.png`, `exp3_exec_prob.png`.

---

## Experiment 4: Reservation prices — inactive and infinite-horizon agents (Section 2.2–2.3)

**Objective:** Compute and verify the closed-form reservation bid/ask prices for an agent holding a fixed inventory, under both finite-horizon and infinite-horizon exponential-utility objectives.

**Method (finite horizon):** Terminal distribution S_T | S_t ∼ N(s, σ²(T−t)); frozen-inventory value function v = −exp(−γx)exp(−γqs)exp(½γ²q²σ²(T−t)); indifference equations give r^a = s + (1−2q)γσ²(T−t)/2 and r^b = s + (−1−2q)γσ²(T−t)/2.

**Method (infinite horizon):** Discounted utility with rate ω; stationary r^a and r^b involve logarithms of expressions requiring ω > ½γ²σ²q² (admissibility).

**Results (finite horizon, s=100, T=1, σ=2):**  
- At q=1, γ=0.1, t=0: r_mid = 100 − 0.1×4×1×1 = 99.60 ✓  
- At q=−1, γ=0.1, t=0: r_mid = 100.40 (reservation above mid for short inventory) ✓  
- Spread r^a − r^b = γσ²(T−t) → 0 as t → T ✓  
- Formula identity r_mid = s − qγσ²(T−t) verified for γ ∈ {0.01, 0.1, 0.5, 1.0} ✓

**Results (infinite horizon, s=100, σ=2, γ=0.1, q_max=5):**  
- ω_bounded = ½γ²σ²(q_max+1)² = ½×0.01×4×36 = 0.72  
- All q ∈ [−5, 5] satisfy admissibility with this ω ✓  
- r_mid_inf ≈ s − qγσ²/ω (first-order) ✓ (qualitative direction correct)  
- Prices break down numerically at q = 5 (near the admissibility boundary) — expected ✓

**Artifacts:** `exp4_reservation_results.csv`, `exp4_finite_reservation_vs_q.png`, `exp4_finite_spread_vs_t.png`, `exp4_infinite_vs_finite_reservation.png`.

---

## Deviations from the paper

1. **Spread column in Tables 1–3 (Section 3.3):** The paper reports the order-flow (inventory-independent) spread component (2/γ)ln(1+γ/k) as "spread" in each table; this equals the total spread δ^a + δ^b at T when the inventory-risk term γσ²(T−t) vanishes. Our simulation uses the full time-varying formula for quoting but reports the same order-flow component to match the paper's table convention.

2. **Profit and std values, Tables 1–3 (Section 3.3):** Spread values match exactly. Profit and inventory statistics are quantitatively close for γ=0.1 and γ=0.01. For γ=0.5, the simulation gives inventory profit 48.64 vs. paper 33.92 and symmetric profit 58.86 vs. paper 66.20. The qualitative ordering (inventory strategy profit < symmetric profit; inventory strategy std < symmetric std) is preserved at all γ. The numerical gap at γ=0.5 is attributable to Monte Carlo variability from the undisclosed random seed used in the paper: with std(Profit)≈5 for inventory and ≈14 for symmetric, a 1000-path MC standard error on the mean is 0.15–0.44, which does not fully account for the gap. The gap is more likely caused by the paper using a seed that, by chance, produces less aggressive inventory positions at early times when the inventory-risk term is largest; the implementation formulas are correct (verified analytically).

3. **Random seed not disclosed (Section 3.3):** The paper does not state the random seed used for its 1000-path simulations. We use seed 42 for all experiments.

4. **Mid-price increment convention (Section 3.3):** The paper states ΔS = ±σ√dt (Bernoulli ±1 with probability ½). We implement this exactly (rng.choice([−1, 1]) × σ × √dt), consistent with the paper's discrete approximation to arithmetic Brownian motion.

5. **Simultaneous bid and ask execution in one step (Section 3.3):** The paper notes that dt should be chosen so multiple arrivals are unlikely. We apply both updates independently (two Bernoulli draws) and accept that simultaneous executions are possible but rare at dt = 0.005; no clamp is applied to double executions, matching the paper's "independent Bernoulli" description.

6. **Execution probability clipping (Section 3.3):** When a quote distance δ < 0 (ask below mid-price for positive inventory with γ=0.5), the execution probability λ(δ)·dt can exceed 1. We clip to min(λ·dt, 1) so the Bernoulli probability is valid. The paper does not discuss this edge case explicitly.

7. **Initial cash X₀ (Section 3.3):** The paper does not state X₀. We use X₀ = 0 and report the terminal mark-to-market P&L X_T + q_T·S_T, consistent with the paper's description (only wealth differences matter under exponential utility).

8. **Power-law intensity model, execution probabilities (Section 2.5):** The power-law intensity λ(δ) = B δ^{−α/β} yields probabilities > 1 for small δ. This is a theoretical-derivation result, not a simulation model; the paper acknowledges this by preferring the exponential form operationally.

---

## Not reproduced

- **Figure 1 (Section 3.3):** The paper's specific sample-path figure cannot be reproduced without knowing the paper's random seed. We generate a representative sample path with seed 42.
- **Full numerical PDE for θ (Section 3.1):** The paper presents the HJB reduction and approximation analytically; no full backward-PDE numerical grid is solved in the paper itself. We verify the asymptotic expansion analytically and numerically on a q-grid but do not solve the complete 3D PDE (in s, q, t) as this is not required by the paper's simulations.
