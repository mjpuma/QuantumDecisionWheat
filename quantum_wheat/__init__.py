"""
quantum_wheat — Quantum decision theory models for agricultural trade networks.

Exports the core simulation function and config factories for convenience:
  from quantum_wheat import run_simulation, make_2country_config, make_5country_config
"""

from .configs import NetworkConfig, make_2country_config, make_5country_config
from .diagnostics import run_simulation

__version__ = "0.2.0"
__all__ = [
    "NetworkConfig",
    "make_2country_config",
    "make_5country_config",
    "run_simulation",
]
