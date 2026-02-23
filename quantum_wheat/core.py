"""
quantum_wheat.core — N-agnostic quantum mechanics for wheat trade networks.

All functions accept n (number of qubits/countries) explicitly. No global variables.
"""

import numpy as np
from scipy.linalg import expm

# Single-qubit Pauli matrices (dtype=complex throughout)
SX = np.array([[0, 1], [1, 0]], dtype=complex)          # σˣ  tunneling
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)       # σʸ  phase
SZ = np.array([[1, 0], [0, -1]], dtype=complex)         # σᶻ  detuning/bias
SP = np.array([[0, 1], [0, 0]], dtype=complex)          # σ⁺  raise to |R⟩
SM = np.array([[0, 0], [1, 0]], dtype=complex)         # σ⁻  lower to |L⟩
I2 = np.eye(2, dtype=complex)
PROJ_R = np.array([[0, 0], [0, 1]], dtype=complex)      # |1⟩⟨1| restrict projector
PROJ_L = np.array([[1, 0], [0, 0]], dtype=complex)      # |0⟩⟨0| liberalize projector


def kron_op(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """
    Embed single-qubit operator `op` at `site` in n-qubit tensor product space.
    Returns a (2^n × 2^n) matrix.
    I ⊗ ... ⊗ op ⊗ ... ⊗ I  with op at position `site`.
    """
    ops = [I2] * n
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def build_hamiltonian(
    delta_t: np.ndarray,
    Delta: np.ndarray,
    J_sym: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Construct the n-country Hamiltonian:
        H = Σᵢ δᵢ σᵢᶻ  +  Σᵢ Δᵢ σᵢˣ  +  Σᵢ<ⱼ J_sym[i,j] σᵢˣ σⱼˣ

    Parameters
    ----------
    delta_t  : (n,) array — time-dependent detuning per country
    Delta    : (n,) array — tunneling (policy flexibility) per country
    J_sym    : (n, n) array — symmetric coupling matrix
    n        : number of qubits (countries)

    Returns (2^n × 2^n) complex Hamiltonian.

    Physical interpretation
    -----------------------
    δᵢ > 0 → biased toward |R⟩ (restrict)
    δᵢ < 0 → biased toward |L⟩ (liberalize)
    Δᵢ     → rate of spontaneous policy reversal (tunneling)
    J_sym  → entanglement strength; for importers this is import dependency;
              for competing exporters this is shared-market overlap fraction.
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        H += delta_t[i] * kron_op(SZ, i, n)
        H += Delta[i] * kron_op(SX, i, n)
    for i in range(n):
        for j in range(i + 1, n):
            if J_sym[i, j] > 0:
                ops = [I2] * n
                ops[i] = SX
                ops[j] = SX
                mat = ops[0]
                for o in ops[1:]:
                    mat = np.kron(mat, o)
                H += J_sym[i, j] * mat
    return H


def lindblad_collapse_operators(
    delta_t: np.ndarray,
    gamma: np.ndarray,
    n: int,
    amp_scale: float = 1.2,
) -> list[np.ndarray]:
    """
    Build Lindblad operators for dephasing + amplitude damping.

    For each country i:
      - Dephasing:  L = sqrt(γᵢ) σᵢᶻ
      - Amplitude damping toward δ-favored state:
          δᵢ > 0 (restrict pressure) → L = sqrt(rate) σᵢ⁻  (collapses toward |R⟩)
          δᵢ < 0 (liberalize pressure) → L = sqrt(rate) σᵢ⁺  (collapses toward |L⟩)
        where rate = amp_scale × γᵢ × (|δᵢ| + 0.1)

    Returns list of (2^n × 2^n) Lindblad operators.
    """
    ops: list[np.ndarray] = []
    for i in range(n):
        ops.append(np.sqrt(gamma[i]) * kron_op(SZ, i, n))
        d = delta_t[i]
        rate = amp_scale * gamma[i] * (abs(d) + 0.1)
        if d > 0:
            ops.append(np.sqrt(rate) * kron_op(SM, i, n))  # SM†=σ⁺ pumps |0⟩→|1⟩: toward |R⟩
        else:
            ops.append(np.sqrt(rate) * kron_op(SP, i, n))  # SP†=σ⁻ pumps |1⟩→|0⟩: toward |L⟩
    return ops


def lindblad_rhs(
    rho: np.ndarray,
    L_ops: list[np.ndarray],
    H: np.ndarray,
) -> np.ndarray:
    """
    Compute RHS of Lindblad master equation:
        dρ/dt = -i[H, ρ] + Σₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})

    Returns dρ/dt as a (2^n × 2^n) complex array.
    """
    drho = -1j * (H @ rho - rho @ H)
    for L in L_ops:
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
    return drho


def schrodinger_evolve(
    rho0: np.ndarray,
    H: np.ndarray,
    dt: float,
    steps: int = 40,
) -> np.ndarray:
    """
    Unitary evolution via matrix exponential over total time dt.
    Uses `steps` sub-steps: U = exp(-iH·dt/steps), ρ → U ρ U†.

    Returns evolved density matrix.
    """
    U = expm(-1j * H * (dt / steps))
    Ud = U.conj().T
    rho = rho0.copy()
    for _ in range(steps):
        rho = U @ rho @ Ud
    return rho


def lindblad_evolve(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    dt: float,
    steps: int = 20,
) -> np.ndarray:
    """
    Euler integration of Lindblad master equation over total time dt.
    Re-normalises trace to 1 after each sub-step to prevent drift.

    Returns evolved density matrix.
    """
    rho = rho0.copy()
    h = dt / steps
    for _ in range(steps):
        rho = rho + h * lindblad_rhs(rho, L_ops, H)
        rho = rho / np.trace(rho).real
    return rho


def restriction_probability(rho: np.ndarray, site: int, n: int) -> float:
    """
    P(restrict)ᵢ = Tr[ρ Πᵢᴿ]
    where Πᵢᴿ = |1⟩⟨1| projects site i onto the restrict state.
    Restrict = spin-up = basis state 1 convention throughout.

    Returns scalar probability in [0, 1].
    """
    Pi = kron_op(PROJ_R, site, n)
    return np.trace(Pi @ rho).real


def reduced_dm(rho: np.ndarray, site: int, n: int) -> np.ndarray:
    """
    Partial trace over all qubits except `site`.
    Returns 2×2 reduced density matrix for country at `site`.
    Uses einsum with explicit index strings for correctness.
    """
    T = rho.reshape([2] * (2 * n))
    alpha = "abcdefghijklmnopqrstuvwxyz"
    in_idx = list(alpha[:n])
    out_idx = list(alpha[n : 2 * n])
    keep_in = in_idx[site]
    keep_out = out_idx[site]
    for k in range(n):
        if k != site:
            out_idx[k] = in_idx[k]
    subscript = "".join(in_idx) + "".join(out_idx) + "->" + keep_in + keep_out
    return np.einsum(subscript, T)


def bloch_vector(rho2: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Bloch vector from 2×2 density matrix.
    Returns (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩).

    Interpretation:
      ⟨σˣ⟩ — quantum coherence (superposition magnitude)
      ⟨σʸ⟩ — relative phase between |R⟩ and |L⟩
      ⟨σᶻ⟩ — restriction bias (+1=fully restrict, -1=fully liberalize)
    """
    return (
        np.trace(SX @ rho2).real,
        np.trace(SY @ rho2).real,
        np.trace(SZ @ rho2).real,
    )


def basis_probs(rho: np.ndarray) -> np.ndarray:
    """
    Return diagonal of density matrix = probabilities over 2^n basis states.
    Index i corresponds to the binary representation of i across n qubits.
    """
    return np.diag(rho).real


def initial_superposition_state(n: int) -> np.ndarray:
    """
    Construct density matrix for |+⟩^⊗n — equal superposition of all basis states.
    This represents maximum policy uncertainty at t=0.
    Returns (2^n × 2^n) density matrix.
    """
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi0 = plus.copy()
    for _ in range(n - 1):
        psi0 = np.kron(psi0, plus)
    return np.outer(psi0, psi0.conj())


def price_signal(
    t: float,
    rng: np.random.Generator,
    crisis_year: float = 10.0,
    crisis_height: float = 2.5,
    crisis_width: float = 1.0,
) -> float:
    """
    Synthetic world wheat price deviation from long-run mean at year t.
    Gaussian crisis spike at crisis_year, mild 6-year commodity cycle baseline.

    Parameters
    ----------
    t             : year (float)
    rng           : numpy random Generator (caller owns seed)
    crisis_year   : year of crisis peak
    crisis_height : peak price deviation (σ units)
    crisis_width  : Gaussian width in years

    Returns scalar price deviation.
    """
    baseline = 0.1 * np.sin(2 * np.pi * t / 6)
    crisis = crisis_height * np.exp(-((t - crisis_year) ** 2) / (2 * crisis_width**2))
    noise = 0.02 * rng.standard_normal()
    return baseline + crisis + noise
