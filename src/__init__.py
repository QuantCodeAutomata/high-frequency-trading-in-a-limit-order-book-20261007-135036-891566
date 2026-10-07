"""
High-frequency trading in a limit order book — Avellaneda & Stoikov (2008).

Core modules:
  config      — simulation and model parameters
  intensity   — execution intensity models (exponential, power-law)
  reservation — reservation price computations (finite & infinite horizon)
  hjb         — HJB framework: value-function transform, theta coefficients
  simulator   — Monte Carlo market-making simulation engine
"""
