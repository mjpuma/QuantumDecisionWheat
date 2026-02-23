"""
Quantum Wheat Network — Wavefunction Visualisation (ported)
==========================================================
Shows the quantum state evolving through the simulation:
  1. Bloch sphere trajectories per country  (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩)
  2. Basis-state probability distribution   |⟨b|Ψ⟩|²  at key moments
  3. Quantum coherence magnitude            |off-diag ρᵢ|  per country
  4. 3D Bloch sphere for Russia and Egypt

Uses quantum_wheat.core and make_5country_config(). Output figures must be
numerically identical to the original quantum_wheat_wavefunctions.py given seed 42.
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quantum_wheat import run_simulation, make_5country_config

warnings.filterwarnings("ignore")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = make_5country_config()
    rng = np.random.default_rng(42)

    result = run_simulation(config, rng)

    prob_history = result["prob_history"]
    bloch_history = result["bloch_history"]
    coherence = result["coherence"]
    basis_snapshots = result["basis_snapshots"]

    N = config.n
    T_YEARS = config.T_years
    LABELS = config.labels
    COLORS = config.colors
    crisis_year = config.crisis_year
    years = np.arange(T_YEARS)
    dim = 2**N

    print("Running wavefunction simulation...")
    print("Done.\n")

    # Figure 1 — Bloch vector components over time
    fig1, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    comp_labels = [
        r"$\langle\sigma^x\rangle$  (quantum coherence)",
        r"$\langle\sigma^y\rangle$  (phase)",
        r"$\langle\sigma^z\rangle$  (restriction bias: +1=restrict, −1=liberalize)"
    ]

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
    plt.close(fig1)

    # Figure 2 — Basis state probability distributions at key moments
    state_labels_short = [format(i, f'0{N}b') for i in range(dim)]

    # Sort by year for consistent left-to-right order (t=0, t=5, t=10, t=11, t=19)
    snapshot_items = sorted(
        basis_snapshots.items(),
        key=lambda kv: int(kv[0].split("=")[1].split(" ")[0])
    )

    fig2, axes2 = plt.subplots(1, len(basis_snapshots), figsize=(16, 5), sharey=True)
    palette = plt.cm.RdYlGn_r

    for ax, (label, probs) in zip(axes2, snapshot_items):
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
    plt.close(fig2)

    # Figure 3 — Quantum coherence magnitude over time
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
    plt.close(fig3)

    # Figure 4 — 3D Bloch sphere trajectory: Russia and Egypt
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
    plt.close(fig4)

    print("All wavefunction figures saved.")

    print("\n── Coherence at crisis vs baseline ─────────────────────────────────")
    for i in range(N):
        base = coherence[:crisis_year, i].mean()
        peak = coherence[crisis_year, i]
        print(f"  {LABELS[i]:10s}  baseline coherence={base:.3f}  crisis={peak:.3f}  Δ={peak-base:+.3f}")


if __name__ == "__main__":
    main()
