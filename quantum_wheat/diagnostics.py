"""
quantum_wheat.diagnostics — Reusable print/plot helpers and simulation runner.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from . import core
from . import configs

if TYPE_CHECKING:
    from .configs import NetworkConfig


def print_matrix(M: np.ndarray, label: str, fmt: str = ".4f") -> None:
    """Pretty-print a complex matrix with a label and row/col alignment."""
    print(f"\n{label}")
    print("-" * (len(label) + 2))
    for i in range(M.shape[0]):
        row_str = "  ".join(
            f"{M[i, j].real:{fmt}}{M[i, j].imag:+{fmt}}j" if M[i, j].imag != 0
            else f"{M[i, j].real:{fmt}}"
            for j in range(M.shape[1])
        )
        print(f"  {row_str}")
    print()


def print_step_workings(
    t_idx: int,
    config: "NetworkConfig",
    rho_pre_sch: np.ndarray,
    rho_post_sch: np.ndarray,
    rho_post_lin: np.ndarray,
    H: np.ndarray,
    U: np.ndarray,
    L_ops: list[np.ndarray],
    delta_t: np.ndarray,
    price: float,
) -> None:
    """
    Print step-by-step quantum workings for one timestep.
    Called from run_2country.py when show_math=True.
    """
    n = config.n
    dim = 2**n
    dt_sch = config.dt_schrodinger
    steps_sch = config.steps_schrodinger

    print("=" * 60)
    print(f"YEAR {t_idx} — STEP-BY-STEP QUANTUM WORKINGS")
    print(f"Price signal p(t) = {price:.4f}")
    print(f"Time-dependent detuning δ(t) = {delta_t}")
    print()

    # [1] Hamiltonian
    evals, evecs = np.linalg.eigh(H)
    print("[1] HAMILTONIAN H ({0}×{0})".format(dim))
    print_matrix(H, "H", ".4f")
    print(f"    Eigenvalues: {evals}")
    print(f"    Eigenvectors:\n{evecs}")
    print("    Physical interpretation:")
    print("      H diagonal: on-site energies (δ bias)")
    print("      H off-diagonal: tunneling (Δ) + coupling (J)")
    print("      off-diagonal structure: entanglement between countries")
    print()

    # [2] Schrödinger evolution
    purity_before = np.trace(rho_pre_sch @ rho_pre_sch).real
    purity_after = np.trace(rho_post_sch @ rho_post_sch).real
    trace_after = np.trace(rho_post_sch).real
    print("[2] SCHRÖDINGER EVOLUTION")
    print(f"    U = exp(-iH·dt/steps)  [dt={dt_sch}, steps={steps_sch}]")
    print_matrix(U, "U", ".4f")
    print("    ρ before:")
    print_matrix(rho_pre_sch, "ρ_pre", ".4f")
    print("    ρ after:")
    print_matrix(rho_post_sch, "ρ_post", ".4f")
    print(f"    Trace check: {trace_after:.6f} (must = 1.0)")
    print(f"    Purity Tr(ρ²) before → after: {purity_before:.4f} → {purity_after:.4f}")
    print()

    # [3] Lindblad operators
    print(f"[3] LINDBLAD OPERATORS ({len(L_ops)} operators)")
    for k, L in enumerate(L_ops):
        norm_L = np.linalg.norm(L)
        print(f"    L_{k}: norm={norm_L:.4f}")
        print_matrix(L, f"L_{k}", ".4f")
    print()

    # [4] Lindblad evolution
    purity_before_lin = np.trace(rho_post_sch @ rho_post_sch).real
    purity_after_lin = np.trace(rho_post_lin @ rho_post_lin).real
    trace_lin = np.trace(rho_post_lin).real
    print("[4] LINDBLAD EVOLUTION")
    print("    ρ before:")
    print_matrix(rho_post_sch, "ρ_pre_lin", ".4f")
    print("    ρ after:")
    print_matrix(rho_post_lin, "ρ_post_lin", ".4f")
    print(f"    Trace check: {trace_lin:.6f}")
    print(f"    Purity Tr(ρ²) before → after: {purity_before_lin:.4f} → {purity_after_lin:.4f}")
    print()

    # [5] Observables
    print("[5] OBSERVABLES")
    for i in range(n):
        rho2 = core.reduced_dm(rho_post_lin, i, n)
        bvec = core.bloch_vector(rho2)
        prob = core.restriction_probability(rho_post_lin, i, n)
        coherence = abs(rho2[0, 1])
        print(f"    {config.labels[i]}:")
        print_matrix(rho2, f"    Reduced ρᵢ (2×2)", ".4f")
        print(f"      Bloch vector: (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩) = {bvec}")
        print(f"      P(restrict) = {prob:.4f}")
        print(f"      Coherence |ρ₀₁| = {coherence:.4f}")
    print("=" * 60)


def run_simulation(
    config: "NetworkConfig",
    rng: np.random.Generator,
    store_workings_years: list[int] | None = None,
) -> dict:
    """
    Execute the full alternating Schrödinger/Lindblad simulation.

    Returns dict with keys:
      prob_history    : (T_years, n) — P(restrict) per country per year
      entanglement    : (T_years,)   — system purity Tr(ρ²)
      bloch_history   : (T_years, n, 3) — Bloch vector components
      coherence       : (T_years, n) — |ρ₀₁| off-diagonal magnitude
      price_series    : (T_years,) — price signal
      basis_snapshots : dict[str, ndarray] — basis probabilities at key years (post-Schrödinger)
      rho_history     : list[ndarray] — density matrix at each year (for math printing)
      H_history       : list[ndarray] — Hamiltonians (for math printing)
      U_history       : list[ndarray] — unitaries (for math printing)
      L_history       : list[list[ndarray]] — Lindblad ops (for math printing)
      delta_history   : (T_years, n) — time-dependent detuning
    """
    n = config.n
    T_years = config.T_years
    years = np.arange(T_years)

    # Key years for basis snapshots (post-Schrödinger)
    key_years = sorted(set([0, 5, config.crisis_year, config.crisis_year + 1, T_years - 1]))
    key_years = [y for y in key_years if 0 <= y < T_years]

    prob_history = np.zeros((T_years, n))
    entanglement = np.zeros(T_years)
    bloch_history = np.zeros((T_years, n, 3))
    coherence = np.zeros((T_years, n))
    delta_history = np.zeros((T_years, n))

    price_series = np.array([
        core.price_signal(
            float(t),
            rng,
            crisis_year=float(config.crisis_year),
            crisis_height=2.5,
            crisis_width=1.0,
        )
        for t in years
    ])

    basis_snapshots: dict[str, np.ndarray] = {}
    rho_history: list[np.ndarray] = []
    H_history: list[np.ndarray] = []
    U_history: list[np.ndarray] = []
    L_history: list[list[np.ndarray]] = []
    step_workings: dict[int, dict] = {}
    store_years = set(store_workings_years or [])

    rho = core.initial_superposition_state(n)

    for t_idx, t in enumerate(years):
        p_t = price_series[t_idx]
        delta_t = config.delta_base + config.price_sensitivity * p_t
        delta_history[t_idx] = delta_t

        H = core.build_hamiltonian(
            delta_t, config.Delta, config.J_sym, n
        )
        L_ops = core.lindblad_collapse_operators(
            delta_t, config.gamma, n, amp_scale=1.2
        )

        rho_pre_sch = rho.copy()
        purity_pre = np.trace(rho_pre_sch @ rho_pre_sch).real

        rho = core.schrodinger_evolve(
            rho, H,
            dt=config.dt_schrodinger,
            steps=config.steps_schrodinger,
        )
        rho_post_sch = rho.copy()
        purity_post_sch = np.trace(rho_post_sch @ rho_post_sch).real

        if purity_post_sch > purity_pre + 1e-10:
            warnings.warn(
                f"Year {t_idx}: Purity increased after Schrödinger "
                f"({purity_pre:.6f} → {purity_post_sch:.6f})"
            )

        if t_idx in key_years:
            basis_snapshots[f"t={t_idx} (post-Schr.)"] = core.basis_probs(rho.copy())

        U = expm(-1j * H * (config.dt_schrodinger / config.steps_schrodinger))

        rho = core.lindblad_evolve(
            rho, H, L_ops,
            dt=config.dt_lindblad,
            steps=config.steps_lindblad,
        )
        rho_post_lin = rho.copy()

        trace_check = abs(np.trace(rho).real - 1.0)
        if trace_check > 1e-6:
            warnings.warn(
                f"Year {t_idx}: Lindblad trace drift |Tr(ρ)-1| = {trace_check:.2e}"
            )

        evals = np.linalg.eigvalsh(rho)
        if np.any(evals < -1e-8):
            warnings.warn(
                f"Year {t_idx}: Negative eigenvalue(s) in ρ: min={evals.min():.2e}"
            )

        purity_post_lin = np.trace(rho_post_lin @ rho_post_lin).real
        if purity_post_lin > purity_post_sch + 1e-10:
            warnings.warn(
                f"Year {t_idx}: Purity increased after Lindblad "
                f"({purity_post_sch:.6f} → {purity_post_lin:.6f})"
            )

        for i in range(n):
            prob_history[t_idx, i] = core.restriction_probability(rho, i, n)
            rho2 = core.reduced_dm(rho, i, n)
            bloch_history[t_idx, i] = core.bloch_vector(rho2)
            coherence[t_idx, i] = abs(rho2[0, 1])

        entanglement[t_idx] = np.trace(rho @ rho).real

        if t_idx in store_years:
            step_workings[t_idx] = {
                "rho_pre_sch": rho_pre_sch.copy(),
                "rho_post_sch": rho_post_sch.copy(),
                "rho_post_lin": rho_post_lin.copy(),
                "H": H.copy(),
                "U": U.copy(),
                "L_ops": [L.copy() for L in L_ops],
                "delta_t": delta_t.copy(),
                "price": float(p_t),
            }

        rho_history.append(rho.copy())
        H_history.append(H.copy())
        U_history.append(U.copy())
        L_history.append([L.copy() for L in L_ops])

    result: dict = {
        "prob_history": prob_history,
        "entanglement": entanglement,
        "bloch_history": bloch_history,
        "coherence": coherence,
        "price_series": price_series,
        "basis_snapshots": basis_snapshots,
        "rho_history": rho_history,
        "H_history": H_history,
        "U_history": U_history,
        "L_history": L_history,
        "delta_history": delta_history,
    }
    if step_workings:
        result["step_workings"] = step_workings
    return result
