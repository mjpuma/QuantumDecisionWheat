"""
Quantum Wheat Network — Wavefunction Visualisation
===================================================
Shows the quantum state evolving through the simulation:
  1. Bloch sphere trajectories per country  (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩)
  2. Basis-state probability distribution   |⟨b|Ψ⟩|²  at key moments
  3. Quantum coherence magnitude            |off-diag ρᵢ|  per country
  4. 3D Bloch sphere for the two most coupled countries (Russia, Egypt)

Can be run standalone: executes the full simulation and generates all figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/savefig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.linalg import expm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Output directory: project_root/figures/
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── re-run the core simulation so we have rho at every timestep ───────────────
rng = np.random.default_rng(42)

N = 5
T_YEARS = 20
LABELS = ["Russia", "USA", "Egypt", "Tunisia", "Lebanon"]
COLORS = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"]
crisis_year = 10

J = np.zeros((N, N))
J[2, 0] = 0.60;  J[2, 1] = 0.25
J[3, 0] = 0.50;  J[3, 1] = 0.30
J[4, 0] = 0.70;  J[4, 1] = 0.15
J_sym = (J + J.T) / 2

delta_base       = np.array([-0.3, -0.4,  0.2,  0.3,  0.4])
Delta            = np.array([ 0.5,  0.6,  0.3,  0.2,  0.15])
gamma            = np.array([ 0.15, 0.12, 0.20, 0.25, 0.30])
price_sensitivity= np.array([ 0.2,  0.2,  0.9,  1.1,  1.3])

years = np.arange(T_YEARS)


def price_signal(t):
    """
    Synthetic price deviation from long-run mean at year t.
    Combines a baseline commodity cycle, a Gaussian crisis spike at crisis_year,
    and small random noise.
    """
    baseline = 0.15 * np.sin(2 * np.pi * t / 6)
    crisis   = 1.8  * np.exp(-((t - crisis_year)**2) / (2 * 2.0**2))
    noise    = 0.05 * rng.standard_normal()
    return baseline + crisis + noise


def kron_op(op, site, n=N):
    """Embed single-qubit operator `op` at `site` in N-qubit space."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[site] = op
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r


def build_hamiltonian(delta_t):
    """
    Build the N-country Hamiltonian: H = Σᵢ δᵢσᵢᶻ + Σᵢ Δᵢσᵢˣ + Σᵢ<ⱼ Jᵢⱼσᵢˣσⱼˣ.
    """
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H += delta_t[i] * kron_op(sz, i)
        H += Delta[i]   * kron_op(sx, i)
    for i in range(N):
        for j in range(i + 1, N):
            if J_sym[i, j] > 0:
                ops = [I2] * N
                ops[i] = sx
                ops[j] = sx
                mat = ops[0]
                for o in ops[1:]:
                    mat = np.kron(mat, o)
                H += J_sym[i, j] * mat
    return H


def lindblad_collapse_operators():
    """
    Return Lindblad operators Lₖ = √γₖ σₖᶻ for dephasing at each site.
    """
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.sqrt(gamma[i]) * kron_op(sz, i) for i in range(N)]


def lindblad_rhs(rho, L_ops, H):
    """Compute RHS of Lindblad equation: dρ/dt = -i[H,ρ] + Σₖ(LₖρLₖ† - ½{Lₖ†Lₖ,ρ})."""
    drho = -1j * (H @ rho - rho @ H)
    for L in L_ops:
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
    return drho


def schrodinger_evolve(rho0, H, dt, steps=40):
    """Unitary evolution via matrix exponential over dt, using `steps` sub-steps."""
    U = expm(-1j * H * (dt / steps))
    Ud = U.conj().T
    rho = rho0.copy()
    for _ in range(steps):
        rho = U @ rho @ Ud
    return rho


def lindblad_evolve(rho0, H, L_ops, dt, steps=20):
    """Euler integration of the Lindblad master equation over dt."""
    rho = rho0.copy()
    h = dt / steps
    for _ in range(steps):
        rho = rho + h * lindblad_rhs(rho, L_ops, H)
        rho = rho / np.trace(rho).real
    return rho


def reduced_dm(rho, site):
    """
    Partial trace over all qubits except `site` → 2×2 density matrix.
    Uses einsum with explicit index strings for clarity and correctness.
    """
    T = rho.reshape([2] * (2 * N))
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    in_idx  = list(alpha[:N])
    out_idx = list(alpha[N:2*N])
    keep_in  = in_idx[site]
    keep_out = out_idx[site]
    for k in range(N):
        if k != site:
            out_idx[k] = in_idx[k]
    subscript = (''.join(in_idx) + ''.join(out_idx) + '->' + keep_in + keep_out)
    return np.einsum(subscript, T)


def bloch_vector(rho2):
    """Compute Bloch vector (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩) from 2×2 density matrix."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return (np.trace(sx @ rho2).real,
            np.trace(sy @ rho2).real,
            np.trace(sz @ rho2).real)


def basis_probs(rho):
    """Extract diagonal of density matrix = probabilities over 2^N basis states."""
    return np.diag(rho).real


def state_label(idx, n=N):
    """Generate a human-readable ket label for basis state index (e.g. |RLRLL⟩)."""
    bits = format(idx, f'0{n}b')
    parts = []
    for i, b in enumerate(bits):
        parts.append(LABELS[i][0] + ("R" if b == "1" else "L"))
    return "|" + " ".join(parts[:2]) + "\n " + " ".join(parts[2:]) + "⟩"


def run_simulation_and_plot():
    """
    Execute the full alternating Schrödinger/Lindblad simulation and generate
    all wavefunction visualisation figures.
    """
    dim = 2**N
    price_series = np.array([price_signal(t) for t in years])

    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi0 = plus.copy()
    for _ in range(N - 1):
        psi0 = np.kron(psi0, plus)
    rho = np.outer(psi0, psi0.conj())

    L_ops = lindblad_collapse_operators()

    bloch_history = np.zeros((T_YEARS, N, 3))
    coherence     = np.zeros((T_YEARS, N))
    basis_snapshots = {}

    snapshot_years = [0, 5, crisis_year, crisis_year + 1, T_YEARS - 1]

    print("Running wavefunction simulation...")
    for t_idx, t in enumerate(years):
        p_t     = price_series[t_idx]
        delta_t = delta_base + price_sensitivity * p_t
        H       = build_hamiltonian(delta_t)

        rho = schrodinger_evolve(rho, H, dt=0.7, steps=40)

        if t_idx in snapshot_years:
            basis_snapshots[f"t={t_idx} (post-Schr.)"] = basis_probs(rho.copy())

        rho = lindblad_evolve(rho, H, L_ops, dt=0.3, steps=20)

        for i in range(N):
            rho2 = reduced_dm(rho, i)
            bx, by, bz = bloch_vector(rho2)
            bloch_history[t_idx, i] = [bx, by, bz]
            coherence[t_idx, i] = abs(rho2[0, 1])

    print("Done.\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Bloch vector components over time
    # ═══════════════════════════════════════════════════════════════════════════

    fig1, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    comp_labels = [r"$\langle\sigma^x\rangle$  (quantum coherence)",
                   r"$\langle\sigma^y\rangle$  (phase)",
                   r"$\langle\sigma^z\rangle$  (restriction bias: +1=restrict, −1=liberalize)"]

    for ax, comp, cl in zip(axes, range(3), comp_labels):
        for i in range(N):
            ax.plot(years, bloch_history[:, i, comp],
                    color=COLORS[i], lw=2, label=LABELS[i],
                    marker="o", ms=3)
        ax.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_ylabel(cl, fontsize=10)
        ax.legend(fontsize=8, loc="upper left", ncol=N)
        ax.set_xlim(0, T_YEARS - 1)
        ax.set_ylim(-1.05, 1.05)

    axes[0].set_title("Bloch Vector Components — Single-Country Reduced Quantum States\n"
                      "Each country's wavefunction projected onto its own Hilbert space",
                      fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Year")

    for ax in axes:
        ax.annotate("Crisis", xy=(crisis_year, ax.get_ylim()[1] * 0.85),
                    fontsize=8, color="#c0392b", ha="center")

    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "wavefunction_bloch_components.png",
                 dpi=150, bbox_inches="tight")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Basis state probability distributions at key moments
    # ═══════════════════════════════════════════════════════════════════════════

    state_labels_short = [format(i, f'0{N}b') for i in range(dim)]

    fig2, axes2 = plt.subplots(1, len(basis_snapshots), figsize=(16, 5), sharey=True)
    palette = plt.cm.RdYlGn_r

    for ax, (label, probs) in zip(axes2, basis_snapshots.items()):
        colors_bar = [palette(p / max(probs.max(), 0.01)) for p in probs]
        ax.bar(range(dim), probs, color=colors_bar, edgecolor="white", lw=0.3)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Basis state  |b₀b₁b₂b₃b₄⟩\n(0=Liberalize, 1=Restrict)", fontsize=7)
        ax.set_xticks(range(0, dim, 4))
        ax.set_xticklabels([state_labels_short[i] for i in range(0, dim, 4)],
                           rotation=90, fontsize=6)
        ax.tick_params(axis='y', labelsize=8)
        ax.axvline(dim - 1, color="#c0392b", lw=1, ls="--", alpha=0.5)

    axes2[0].set_ylabel("Probability  |⟨b|Ψ⟩|²", fontsize=10)
    fig2.suptitle("Wavefunction Probability Distribution over Basis States  |b₀b₁b₂b₃b₄⟩\n"
                  "b=0: Liberalize  |  b=1: Restrict  |  Rightmost bar = all countries restrict",
                  fontsize=11, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "wavefunction_basis_distribution.png",
                 dpi=150, bbox_inches="tight")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Quantum coherence magnitude over time
    # ═══════════════════════════════════════════════════════════════════════════

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for i in range(N):
        ax3.plot(years, coherence[:, i], color=COLORS[i], lw=2.5,
                 label=LABELS[i], marker="o", ms=4)
    ax3.axvline(crisis_year, color="#e74c3c", lw=1.8, ls=":", alpha=0.8,
                label=f"Crisis (yr {crisis_year})")
    ax3.fill_between(years, coherence[:, 4], alpha=0.08, color=COLORS[4])
    ax3.set_title("Quantum Coherence  |ρ₀₁|  per Country\n"
                  "Magnitude of off-diagonal element of reduced density matrix — "
                  "how much each country is in superposition",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("|ρ₀₁|  (0=classical, 0.5=maximum superposition)")
    ax3.set_ylim(0, 0.55)
    ax3.set_xlim(0, T_YEARS - 1)
    ax3.legend(fontsize=9)
    ax3.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax3.text(T_YEARS - 0.5, 0.51, "max superposition", fontsize=7, color="gray", ha="right")
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "wavefunction_coherence.png",
                 dpi=150, bbox_inches="tight")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 4 — 3D Bloch sphere trajectory: Russia and Egypt
    # ═══════════════════════════════════════════════════════════════════════════

    fig4 = plt.figure(figsize=(14, 6))

    for col, (site, country) in enumerate([(0, "Russia"), (2, "Egypt")]):
        ax = fig4.add_subplot(1, 2, col + 1, projection='3d')

        bx = bloch_history[:, site, 0]
        by = bloch_history[:, site, 1]
        bz = bloch_history[:, site, 2]

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.15, lw=0.4)

        for t in range(T_YEARS - 1):
            c = plt.cm.viridis(t / T_YEARS)
            ax.plot(bx[t:t+2], by[t:t+2], bz[t:t+2], color=c, lw=2.5)

        ax.scatter([bx[0]], [by[0]], [bz[0]], color="green", s=80, zorder=5, label="t=0")
        ax.scatter([bx[crisis_year]], [by[crisis_year]], [bz[crisis_year]],
                   color="red", s=120, marker="*", zorder=5, label=f"Crisis yr {crisis_year}")
        ax.scatter([bx[-1]], [by[-1]], [bz[-1]], color="purple", s=80, zorder=5, label="t=final")

        ax.set_xlabel(r"$\langle\sigma^x\rangle$", fontsize=9)
        ax.set_ylabel(r"$\langle\sigma^y\rangle$", fontsize=9)
        ax.set_zlabel(r"$\langle\sigma^z\rangle$", fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.text(0, 0,  1.15, "|R⟩\nRestrict",  ha="center", fontsize=8, color="#c0392b", fontweight="bold")
        ax.text(0, 0, -1.15, "|L⟩\nLiberalize", ha="center", fontsize=8, color="#2980b9", fontweight="bold")
        ax.set_title(f"{country} — Bloch Sphere Trajectory\n"
                    f"(coloured early→late: purple→yellow)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

    fig4.suptitle("Bloch Sphere: Quantum State Trajectory of Russia and Egypt\n"
                  "The state vector traces the country's decision wavefunction through policy space",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "wavefunction_bloch_sphere.png",
                 dpi=150, bbox_inches="tight")

    print("All wavefunction figures saved.")

    print("\n── Coherence at crisis vs baseline ─────────────────────────────────")
    for i in range(N):
        base = coherence[:crisis_year, i].mean()
        peak = coherence[crisis_year, i]
        print(f"  {LABELS[i]:10s}  baseline coherence={base:.3f}  crisis={peak:.3f}  Δ={peak-base:+.3f}")


if __name__ == "__main__":
    run_simulation_and_plot()
