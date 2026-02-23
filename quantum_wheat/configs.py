"""
quantum_wheat.configs — Network configurations and factory functions.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class NetworkConfig:
    """
    Complete specification of a quantum wheat trade network.

    Attributes
    ----------
    n                  : number of countries (qubits)
    labels             : country names, length n
    colors             : matplotlib hex colours, length n
    J                  : (n, n) coupling matrix — see coupling_type
    coupling_type      : 'import_dependency' or 'market_overlap'
                         import_dependency: J[i,j] = fraction of i's imports from j
                         market_overlap:    J[i,j] = fraction of shared export markets
                         Determines physical interpretation of coupling in methods section.
    delta_base         : (n,) baseline detuning (negative=liberal, positive=restrict)
    Delta              : (n,) tunneling / policy flexibility per country
    gamma              : (n,) decoherence rate (policy announcement frequency)
    price_sensitivity  : (n,) response strength to price signal
    crisis_year        : year index of synthetic crisis peak
    T_years            : total simulation length in years
    dt_schrodinger     : Schrödinger evolution timestep per year (default 0.7)
    dt_lindblad        : Lindblad evolution timestep per year (default 0.3)
    steps_schrodinger  : sub-steps for matrix exponentiation (default 40)
    steps_lindblad     : Euler sub-steps for Lindblad (default 20)
    """

    n: int
    labels: list[str]
    colors: list[str]
    J: np.ndarray
    coupling_type: str  # 'import_dependency' | 'market_overlap'
    delta_base: np.ndarray
    Delta: np.ndarray
    gamma: np.ndarray
    price_sensitivity: np.ndarray
    crisis_year: int
    T_years: int
    dt_schrodinger: float = 0.7
    dt_lindblad: float = 0.3
    steps_schrodinger: int = 40
    steps_lindblad: int = 20

    @property
    def J_sym(self) -> np.ndarray:
        """Symmetrised coupling matrix used in Hamiltonian."""
        return (self.J + self.J.T) / 2

    def summary(self) -> str:
        """Return human-readable parameter summary for print/logging."""
        lines = [
            f"Network: {self.n} countries, {self.coupling_type}",
            f"  Labels: {', '.join(self.labels)}",
            f"  Crisis year: {self.crisis_year}, T_years: {self.T_years}",
            f"  J_sym (sample):\n{np.round(self.J_sym, 3)}",
        ]
        return "\n".join(lines)


def make_2country_config() -> NetworkConfig:
    """
    France–Russia: two competing wheat exporters.

    Coupling mechanism: MARKET OVERLAP (not import dependency).
    J[i,j] represents the fraction of shared MENA export market destinations.
    When Russia restricts, France faces simultaneous gap-filling pressure (liberalize)
    AND domestic price spillover (restrict) — this quantum ambiguity is the
    physically interesting feature absent from classical cascade models.

    Key empirical motivation: 2010-11 Russian export ban.
    Russia banned exports August 2010 (collapses to |R⟩).
    France/EU accelerated exports to fill gap through spring 2011 (collapses to |L⟩).
    Anticorrelated collapse from entangled superposition — the ordering was not
    predetermined; it emerged from the coupled quantum dynamics.

    Parameters chosen to reflect:
      - Both are net exporters → negative delta_base (liberal baseline)
      - France more liberal baseline under EU Common Agricultural Policy (CAP)
        which uses export licenses rather than outright bans
      - France higher Delta: CAP mechanism allows faster policy reversal
      - Russia lower gamma: fewer policy announcements per year (more unitary evolution)
      - Russia higher price_sensitivity: historically more reactive to export price spikes
      - crisis_year=5, T_years=12: compact for pedagogical step-through

    J matrix:
      J[Russia, France] = 0.30  — Russia affected by France competing in same markets
      J[France, Russia] = 0.35  — France more exposed (smaller exporter, more sensitive
                                   to Russian supply shocks)
    """
    n = 2
    J = np.zeros((n, n))
    J[0, 1] = 0.30  # Russia ← France market competition
    J[1, 0] = 0.35  # France ← Russia market competition (asymmetric: France more exposed)

    return NetworkConfig(
        n=n,
        labels=["Russia", "France"],
        colors=["#c0392b", "#002395"],  # Russian red, French blue
        J=J,
        coupling_type="market_overlap",
        delta_base=np.array([-0.3, -0.5]),  # both liberal; France more so (EU CAP)
        Delta=np.array([0.4, 0.6]),  # France higher flexibility
        gamma=np.array([0.15, 0.10]),  # Russia announces less frequently
        price_sensitivity=np.array([0.60, 0.45]),  # Russia more price-reactive
        crisis_year=5,
        T_years=12,
    )


def make_5country_config() -> NetworkConfig:
    """
    Russia, USA (exporters) + Egypt, Tunisia, Lebanon (importers).

    Coupling mechanism: IMPORT DEPENDENCY.
    J[i,j] = fraction of country i's wheat imports sourced from country j.
    This is a supply-chain coupling: importers' policy states are entangled
    with exporters' through trade dependency.

    Parameters are identical to the original quantum_wheat_synthetic.py.
    Output figures must be numerically identical given the same rng seed (42).

    J matrix (import dependency):
      Egypt:   60% Russia, 25% USA
      Tunisia: 50% Russia, 30% USA
      Lebanon: 70% Russia, 15% USA
      Exporters have zero import dependency in this model.
    """
    n = 5
    J = np.zeros((n, n))
    J[2, 0] = 0.60
    J[2, 1] = 0.25  # Egypt
    J[3, 0] = 0.50
    J[3, 1] = 0.30  # Tunisia
    J[4, 0] = 0.70
    J[4, 1] = 0.15  # Lebanon

    return NetworkConfig(
        n=n,
        labels=["Russia", "USA", "Egypt", "Tunisia", "Lebanon"],
        colors=["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"],
        J=J,
        coupling_type="import_dependency",
        delta_base=np.array([-0.3, -0.4, 0.2, 0.3, 0.4]),
        Delta=np.array([0.5, 0.6, 0.3, 0.2, 0.15]),
        gamma=np.array([0.15, 0.12, 0.20, 0.25, 0.30]),
        price_sensitivity=np.array([0.55, 0.50, 1.2, 1.4, 1.6]),
        crisis_year=10,
        T_years=20,
    )
