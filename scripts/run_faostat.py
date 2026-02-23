"""
Quantum Wheat Trade Network — FAOSTAT Real Data Runner (stub)
=============================================================
Placeholder for N-country simulation driven by real FAO trade data.

Expected CSV format (from FAOSTAT Detailed Trade Matrix):
  reporter_country, partner_country, year, value_1000USD, quantity_tonnes

The load_faostat_data function will:
  1. Infer N from unique reporter countries
  2. Build J matrix from bilateral trade flows (import dependency ratios)
  3. Estimate delta_base from historical restriction frequency (AMAD/IFPRI data)
  4. Return a NetworkConfig ready for run_simulation()
"""

from __future__ import annotations

from quantum_wheat.configs import NetworkConfig


def load_faostat_data(filepath: str, crop: str = "wheat") -> NetworkConfig:
    """
    Load FAOSTAT trade data and build a NetworkConfig.

    Parameters
    ----------
    filepath : path to FAOSTAT Detailed Trade Matrix CSV
    crop     : commodity filter (default 'wheat')

    Returns NetworkConfig with coupling_type='import_dependency'.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "FAOSTAT data loading not yet implemented.\n"
        "Expected input: FAOSTAT Detailed Trade Matrix CSV with columns\n"
        "  [reporter_country, partner_country, year, value_1000USD, quantity_tonnes]\n"
        "Download from: https://www.fao.org/faostat/en/#data/TM"
    )
