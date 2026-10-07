#!/usr/bin/env bash
# reproduce.sh — run all experiments for
# Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"

set -u

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo "Reproducing Avellaneda & Stoikov (2008)"
echo "High-frequency trading in a limit order book"
echo "============================================================"
echo ""

# 1. Unit tests
echo ">>> Running unit tests ..."
python -m pytest tests/test_src.py -v --tb=short
echo ""

# 2. Experiment 1: Monte Carlo simulation (Tables 1–3)
echo ">>> Experiment 1: Inventory vs Symmetric strategy (Monte Carlo)"
python exp/exp1_simulation.py
echo ""

# 3. Experiment 2: HJB analytical derivation and verification
echo ">>> Experiment 2: HJB framework and asymptotic expansion"
python exp/exp2_hjb.py
echo ""

# 4. Experiment 3: Execution intensity derivation
echo ">>> Experiment 3: Order-execution intensity functions"
python exp/exp3_intensity.py
echo ""

# 5. Experiment 4: Reservation prices (finite and infinite horizon)
echo ">>> Experiment 4: Reservation prices for inactive/infinite-horizon agents"
python exp/exp4_reservation.py
echo ""

echo "============================================================"
echo "All experiments complete.  Results saved in results/"
echo "============================================================"
