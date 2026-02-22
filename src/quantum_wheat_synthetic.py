"""
Quantum Wheat Trade Network Model — Synthetic Data Demonstration
================================================================
Implements the alternating Schrödinger / Lindblad simulation for a small
network of wheat-trading nations.

Network (synthetic):
  Exporters: Russia (0), USA (1)
  Importers:  Egypt (2), Tunisia (3), Lebanon (4)

Pipeline per annual timestep
  1. Schrödinger evolution: coherent build-up of joint decision state
  2. Lindblad decoherence:  policy announcements collapse superpositions
  3. Record observables, advance price signal, repeat

Author: M.J. Puma et al.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/savefig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import expm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Output directory: project_root/figures/
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NETWORK PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

N = 5          # number of countries
T_YEARS = 20   # simulation length (years)
LABELS = ["Russia", "USA", "Egypt", "Tunisia", "Lebanon"]
COLORS = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"]

# --- Coupling matrix J (import dependency, asymmetric) -----------------------
# J[i,j] = fraction of country i's wheat imports sourced from country j
# Exporters (0,1) have zero import dependency on others in this toy model
J = np.zeros((N, N))

# Egypt: 60% Russia, 25% USA
J[2, 0] = 0.60;  J[2, 1] = 0.25

# Tunisia: 50% Russia, 30% USA
J[3, 0] = 0.50;  J[3, 1] = 0.30

# Lebanon: 70% Russia, 15% USA
J[4, 0] = 0.70;  J[4, 1] = 0.15

# Symmetric coupling term used in Hamiltonian:  J_sym[i,j] = (J[i,j]+J[j,i])/2
J_sym = (J + J.T) / 2

print("Coupling matrix J (import dependency):")
print(np.round(J, 2))
print()

# --- Baseline detuning δ (domestic bias toward restriction) ------------------
# Positive  → biased toward |R⟩  (restrict)
# Negative  → biased toward |L⟩  (liberalize)
delta_base = np.array([
    -0.3,   # Russia  — large exporter, baseline liberal
    -0.4,   # USA     — large exporter, baseline liberal
     0.2,   # Egypt   — moderate import pressure
     0.3,   # Tunisia — higher vulnerability
     0.4,   # Lebanon — most vulnerable
])

# --- Tunneling Δ (policy flexibility) ----------------------------------------
Delta = np.array([0.5, 0.6, 0.3, 0.2, 0.15])

# --- Decoherence rates γ (announcement frequency, per year) ------------------
# Lower rates let Schrödinger evolution dominate between announcements
gamma = np.array([0.15, 0.12, 0.20, 0.25, 0.30])


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC PRICE SIGNAL
#     Produces a smooth price series with a crisis spike at year 10
# ═══════════════════════════════════════════════════════════════════════════════

years = np.arange(T_YEARS)

def price_signal(t, crisis_year=10, crisis_height=1.8, crisis_width=2.0):
    """Normalised price deviation from long-run mean (0 = average)."""
    baseline = 0.15 * np.sin(2 * np.pi * t / 6)          # ~6-year commodity cycle
    crisis   = crisis_height * np.exp(-((t - crisis_year)**2) / (2 * crisis_width**2))
    noise    = 0.05 * rng.standard_normal()
    return baseline + crisis + noise

price_series = np.array([price_signal(t) for t in years])

# Map price to country-specific detuning: δᵢ(t) = δ_base_i + wᵢ × p(t)
# Importers are more sensitive to price spikes than exporters
price_sensitivity = np.array([0.2, 0.2, 0.9, 1.1, 1.3])


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  QUANTUM OPERATORS  (single-qubit Pauli matrices, N-qubit extensions)
# ═══════════════════════════════════════════════════════════════════════════════

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)          # σˣ  (tunneling)
sz = np.array([[1, 0], [0, -1]], dtype=complex)         # σᶻ  (detuning)
sp = np.array([[0, 1], [0, 0]], dtype=complex)          # σ⁺  (raise to |R⟩)
sm = np.array([[0, 0], [1, 0]], dtype=complex)          # σ⁻  (lower to |L⟩)

dim = 2**N   # Hilbert space dimension (32 for N=5)


def kron_op(op, site, n=N):
    """Embed single-qubit operator `op` at `site` in N-qubit space."""
    ops = [I2] * n
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def build_hamiltonian(delta_t):
    """
    H = Σᵢ δᵢ σᵢᶻ  +  Σᵢ Δᵢ σᵢˣ  +  Σᵢ<ⱼ Jᵢⱼ σᵢˣ σⱼˣ
    """
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N):
        H += delta_t[i] * kron_op(sz, i)
        H += Delta[i]   * kron_op(sx, i)

    for i in range(N):
        for j in range(i + 1, N):
            if J_sym[i, j] > 0:
                # σᵢˣ ⊗ σⱼˣ  coupling
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
    Lindblad operators Lₖ = √γₖ σₖᶻ for dephasing at each site.
    Dephasing suppresses off-diagonal coherence (phase between |L⟩ and |R⟩).
    """
    ops = []
    for i in range(N):
        ops.append(np.sqrt(gamma[i]) * kron_op(sz, i))   # dephasing
    return ops


def lindblad_rhs(rho, L_ops, H):
    """dρ/dt = -i[H,ρ] + Σₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})"""
    drho = -1j * (H @ rho - rho @ H)
    for L in L_ops:
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
    return drho


def schrodinger_evolve(rho0, H, dt, steps=20):
    """Unitary evolution via matrix exponential, dt/steps sub-steps."""
    U = expm(-1j * H * (dt / steps))
    Ud = U.conj().T
    rho = rho0.copy()
    for _ in range(steps):
        rho = U @ rho @ Ud
    return rho


def lindblad_evolve(rho0, H, L_ops, dt, steps=20):
    """Simple Euler integration of Lindblad equation."""
    rho = rho0.copy()
    h = dt / steps
    for _ in range(steps):
        rho = rho + h * lindblad_rhs(rho, L_ops, H)
        # renormalise to keep trace = 1
        rho = rho / np.trace(rho).real
    return rho


def restriction_probability(rho, site):
    """
    P(restrict)ᵢ = Tr[ ρ  Πᵢ^R ]
    Πᵢ^R projects site i onto |1⟩ (restrict = spin-up convention).
    """
    proj_up = np.array([[0, 0], [0, 1]], dtype=complex)   # |1⟩⟨1| projects onto restrict
    Pi = kron_op(proj_up, site)
    return np.trace(Pi @ rho).real


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  INITIAL STATE  — all countries in equal superposition |+⟩^⊗N
# ═══════════════════════════════════════════════════════════════════════════════

plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
psi0 = plus.copy()
for _ in range(N - 1):
    psi0 = np.kron(psi0, plus)

rho = np.outer(psi0, psi0.conj())   # pure state density matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SIMULATION LOOP
#     Each year:
#       (a) compute time-dependent Hamiltonian from price signal
#       (b) Schrödinger phase  (0.5 yr): coherent build-up
#       (c) Lindblad phase     (0.5 yr): announcement / decoherence
#       (d) record restriction probabilities
# ═══════════════════════════════════════════════════════════════════════════════

L_ops = lindblad_collapse_operators()

prob_history  = np.zeros((T_YEARS, N))   # P(restrict) per country per year
entanglement  = np.zeros(T_YEARS)        # purity Tr(ρ²) as proxy

print("Running simulation...")

for t_idx, t in enumerate(years):
    p_t = price_series[t_idx]
    delta_t = delta_base + price_sensitivity * p_t

    H = build_hamiltonian(delta_t)

    # (b) Schrödinger — coherent half-year
    rho = schrodinger_evolve(rho, H, dt=0.7, steps=40)

    # (c) Lindblad — decoherence half-year (announcement season)
    rho = lindblad_evolve(rho, H, L_ops, dt=0.3, steps=20)

    # (d) Observables
    for i in range(N):
        prob_history[t_idx, i] = restriction_probability(rho, i)

    entanglement[t_idx] = np.trace(rho @ rho).real

print("Simulation complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 16))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

crisis_year = 10

# ── 6a. Price signal ──────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.fill_between(years, price_series, alpha=0.3, color="#e74c3c")
ax0.plot(years, price_series, color="#c0392b", lw=2)
ax0.axhline(0, color="gray", lw=0.8, ls="--")
ax0.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.7,
            label=f"Crisis peak (year {crisis_year})")
ax0.set_title("Synthetic World Wheat Price Signal  (deviation from long-run mean)",
              fontsize=12, fontweight="bold")
ax0.set_xlabel("Year")
ax0.set_ylabel("Price deviation (σ)")
ax0.legend(fontsize=9)
ax0.set_xlim(0, T_YEARS - 1)

# ── 6b. Restriction probabilities — exporters ─────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
for i in [0, 1]:
    ax1.plot(years, prob_history[:, i], color=COLORS[i],
             lw=2, label=LABELS[i], marker="o", ms=4)
ax1.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
ax1.set_title("Exporters: P(Restrict)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Year");  ax1.set_ylabel("P(restrict)")
ax1.set_ylim(0, 1);      ax1.legend(fontsize=9)
ax1.set_xlim(0, T_YEARS - 1)

# ── 6c. Restriction probabilities — importers ─────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
for i in [2, 3, 4]:
    ax2.plot(years, prob_history[:, i], color=COLORS[i],
             lw=2, label=LABELS[i], marker="o", ms=4)
ax2.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
ax2.set_title("Importers: P(Restrict / Hoard)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Year");  ax2.set_ylabel("P(restrict)")
ax2.set_ylim(0, 1);      ax2.legend(fontsize=9)
ax2.set_xlim(0, T_YEARS - 1)

# ── 6d. System purity Tr(ρ²)  — proxy for quantum coherence ──────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(years, entanglement, color="#2c3e50", lw=2)
ax3.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
ax3.fill_between(years, entanglement, alpha=0.15, color="#2c3e50")
ax3.set_title("System Purity  Tr(ρ²)\n(1 = pure, <1 = decoherence / classical)",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Year");  ax3.set_ylabel("Tr(ρ²)")
ax3.set_ylim(0, 1.05)
ax3.set_xlim(0, T_YEARS - 1)

# ── 6e. Phase space: Russia vs Egypt ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
sc = ax4.scatter(prob_history[:, 0], prob_history[:, 2],
                 c=years, cmap="viridis", s=60, zorder=3)
ax4.plot(prob_history[:, 0], prob_history[:, 2],
         color="gray", lw=0.8, alpha=0.5, zorder=2)
# mark crisis year
cx, cy = prob_history[crisis_year, 0], prob_history[crisis_year, 2]
ax4.scatter([cx], [cy], color="#e74c3c", s=150, zorder=5,
            marker="*", label=f"Crisis (yr {crisis_year})")
plt.colorbar(sc, ax=ax4, label="Year")
ax4.set_xlabel("P(Russia restricts)");  ax4.set_ylabel("P(Egypt restricts)")
ax4.set_title("Phase Space: Russia vs Egypt", fontsize=11, fontweight="bold")
ax4.set_xlim(0, 1);  ax4.set_ylim(0, 1)
ax4.legend(fontsize=9)

# ── 6f. Network correlation heatmap (post-crisis vs baseline) ─────────────────
ax5 = fig.add_subplot(gs[3, :])
baseline  = prob_history[:crisis_year, :]
post      = prob_history[crisis_year:, :]

corr_base = np.corrcoef(baseline.T)
corr_post = np.corrcoef(post.T)

# Side-by-side correlation matrices
combined = np.hstack([corr_base, np.full((N, 1), np.nan), corr_post])
im = ax5.imshow(combined, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax5, fraction=0.02)

ticks = list(range(N)) + [N] + list(range(N + 1, 2 * N + 1))
tick_labels = LABELS + [""] + LABELS
ax5.set_xticks(ticks);  ax5.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
ax5.set_yticks(range(N));  ax5.set_yticklabels(LABELS, fontsize=8)
ax5.set_title(
    "Policy Correlation Matrix:  Baseline (years 0–9, left)  vs  Post-Crisis (years 10–19, right)\n"
    "Blue = anticorrelated  |  Red = correlated",
    fontsize=11, fontweight="bold"
)

# Divider
ax5.axvline(N - 0.5, color="black", lw=2)
ax5.text(N / 2 - 0.5, -1.0, "BASELINE", ha="center", fontsize=9, fontweight="bold")
ax5.text(N + N / 2 + 0.5, -1.0, "POST-CRISIS", ha="center", fontsize=9, fontweight="bold")

fig.suptitle(
    "Quantum Wheat Trade Network  —  Synthetic Demonstration\n"
    "Alternating Schrödinger / Lindblad Simulation  |  N=5 countries, T=20 years",
    fontsize=13, fontweight="bold", y=0.98
)

out_path = FIGURES_DIR / "quantum_wheat_synthetic.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PRINT KEY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Crisis response (year 10) ──────────────────────────────────────")
for i in range(N):
    pre  = prob_history[crisis_year - 1, i]
    peak = prob_history[crisis_year, i]
    print(f"  {LABELS[i]:10s}  pre={pre:.3f}  peak={peak:.3f}  Δ={peak-pre:+.3f}")

print("\n── Correlation shift (baseline vs post-crisis) ──────────────────")
pairs = [(0, 2, "Russia–Egypt"), (0, 3, "Russia–Tunisia"),
         (0, 4, "Russia–Lebanon"), (2, 3, "Egypt–Tunisia")]
for i, j, name in pairs:
    b = np.corrcoef(baseline[:, i], baseline[:, j])[0, 1]
    p = np.corrcoef(post[:, i],     post[:, j])[0, 1]
    print(f"  {name:20s}  baseline={b:+.3f}  post-crisis={p:+.3f}  Δ={p-b:+.3f}")
