# High-frequency trading in a limit order book

**Authors:** Marco Avellaneda & Sasha Stoikov  
**Published:** *Quantitative Finance*, 8(3), 217–224, 2008  
**Preprint date:** October 5, 2006

---

## Overview

This repository reproduces the four main experiments from Avellaneda & Stoikov (2008), which derives and validates an inventory-based market-making strategy in a limit order book.  The paper models the dealer's problem under exponential utility, derives approximate closed-form bid/ask quotes from a HJB reduction, calibrates execution intensities from microstructure assumptions, and validates the resulting strategy via Monte Carlo simulation.

## Repository structure

```
├── src/               # Core library
│   ├── config.py          # SimParams dataclass with paper parameters
│   ├── intensity.py       # Exponential and power-law intensity models λ(δ)
│   ├── reservation.py     # Finite- and infinite-horizon reservation prices
│   ├── hjb.py             # HJB transform, asymptotic expansion, first-order conditions
│   └── simulator.py       # Monte Carlo market-making simulator
├── exp/               # Experiment scripts (call src/ only)
│   ├── exp1_simulation.py # Table 1–3 Monte Carlo comparison (Section 3.3)
│   ├── exp2_hjb.py        # HJB framework derivation & verification (Section 3.1–3.2)
│   ├── exp3_intensity.py  # Execution intensity derivation (Section 2.5)
│   └── exp4_reservation.py# Reservation prices: inactive & infinite-horizon (Section 2.2–2.3)
├── tests/
│   └── test_src.py        # 54 pytest tests for all src/ modules
├── results/           # Saved metrics, tables, and plots
│   └── RESULTS.md         # Experiment results vs. paper values
├── data/              # Data directory (no external data needed; simulations only)
├── reproduce.sh       # One-command reproduction entry point
├── .env.example       # Environment variable template (MASSIVE_API_KEY not required)
└── requirements.txt
```

## Quick start

```bash
# 1. Clone and enter repository
git clone <repo-url> && cd <repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all experiments
bash reproduce.sh
```

## Parameters (Section 3.3)

| Symbol | Value | Description |
|--------|-------|-------------|
| s₀ | 100 | Initial mid-price |
| T | 1.0 | Horizon (trading day = 1) |
| σ | 2.0 | Mid-price volatility |
| dt | 0.005 | Time step (N = 200 steps) |
| k | 1.5 | Intensity decay parameter |
| A | 140 | Intensity scale parameter |
| n\_paths | 1000 | Monte Carlo paths per γ |
| γ | {0.1, 0.01, 0.5} | Risk-aversion grid |

## Key results (Section 3.3, Tables 1–3)

The inventory-based strategy produces lower P&L variance and lower final-inventory variance than the symmetric benchmark at every tested γ level, confirming the paper's main claim.

| γ | Strategy | Spread | Profit | std(Profit) | std(Final q) |
|---|----------|--------|--------|-------------|--------------|
| 0.1 | Inventory | 1.29 | 65.01 | 6.77 | 3.01 |
| 0.1 | Symmetric | 1.29 | 68.65 | 12.47 | 8.41 |
| 0.01 | Inventory | 1.33 | 68.70 | 9.12 | 5.22 |
| 0.01 | Symmetric | 1.33 | 69.04 | 13.11 | 8.72 |
| 0.5 | Inventory | 1.15 | 48.64 | 5.83 | 2.00 |
| 0.5 | Symmetric | 1.15 | 58.86 | 11.11 | 6.99 |

See `results/RESULTS.md` for a full paper-vs-simulation comparison.

## Running tests

```bash
pytest tests/test_src.py -v
```

All 54 tests should pass.
