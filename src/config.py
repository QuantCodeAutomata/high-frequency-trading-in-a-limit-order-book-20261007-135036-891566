"""
Model parameters for Avellaneda & Stoikov (2008).

All values taken directly from Section 3.3 "Numerical simulations"
of the paper, pp. 10–11.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SimParams:
    """Parameters for the Monte Carlo simulation (Section 3.3)."""

    # Mid-price and horizon
    s0: float = 100.0     # Initial mid-price
    T: float = 1.0        # Horizon (dimensionless; one trading day)
    sigma: float = 2.0    # Volatility of mid-price (arithmetic Brownian)
    dt: float = 0.005     # Time step (N = T/dt = 200 steps)

    # Inventory and utility
    q0: int = 0           # Initial inventory
    gamma: float = 0.1    # Risk-aversion coefficient

    # Intensity parameters (eq. 2.8 + 2.11, p. 6-7)
    k: float = 1.5        # Exponential decay of execution intensity
    A: float = 140.0      # Scale of execution intensity

    # Simulation
    n_paths: int = 1000   # Monte Carlo paths
    seed: int = 42        # Random seed for reproducibility

    @property
    def N(self) -> int:
        """Number of time steps."""
        return round(self.T / self.dt)

    @property
    def t_grid(self):
        """Time grid 0, dt, 2dt, …, T."""
        import numpy as np
        return np.linspace(0.0, self.T, self.N + 1)


# Gamma values explored in Tables 1–3 (Section 3.3)
GAMMA_VALUES: List[float] = [0.1, 0.01, 0.5]

# Default parameters (γ = 0.1, Table 1)
DEFAULT_PARAMS = SimParams()

# Parameters for infinite-horizon reservation-price experiments (Section 2.3)
INF_HORIZON_QMAX: int = 5   # Example inventory bound for illustration
