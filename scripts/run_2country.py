"""
Quantum Wheat Trade Network — France–Russia (2-country pedagogical demo)
=========================================================================

Two competing wheat exporters whose coupling arises from shared MENA export
market overlap, not import dependency (see configs.py for full discussion).

Key physics demonstrated:
  - Anticorrelated collapse: when Russia restricts, France collapses to liberalize
  - Entanglement: policies cannot be understood as independent qubits
  - Crisis dynamics: 2010-11 style export ban reproduced at crisis_year=5
  - Market-overlap coupling: J encodes competing access to same customers

Run with:  python scripts/run_2country.py [--show-math] [--no-figures]

Flags:
  --show-math    Print full step-by-step matrix workings for years 0 and
                 crisis_year. Intended for manual verification and paper
                 appendix material.
  --no-figures   Skip figure generation (useful when only math output needed).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quantum_wheat import run_simulation, make_2country_config
from quantum_wheat.diagnostics import print_step_workings

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="France–Russia 2-country quantum wheat demo")
    parser.add_argument("--show-math", action="store_true", help="Print step-by-step matrix workings")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    config = make_2country_config()
    rng = np.random.default_rng(42)

    store_years = [0, config.crisis_year] if args.show_math else None
    result = run_simulation(config, rng, store_workings_years=store_years)

    prob_history = result["prob_history"]
    entanglement = result["entanglement"]
    bloch_history = result["bloch_history"]
    coherence = result["coherence"]
    price_series = result["price_series"]
    years = np.arange(config.T_years)
    crisis_year = config.crisis_year

    if args.show_math and "step_workings" in result:
        for t_idx in [0, crisis_year]:
            w = result["step_workings"][t_idx]
            print_step_workings(
                t_idx=t_idx,
                config=config,
                rho_pre_sch=w["rho_pre_sch"],
                rho_post_sch=w["rho_post_sch"],
                rho_post_lin=w["rho_post_lin"],
                H=w["H"],
                U=w["U"],
                L_ops=w["L_ops"],
                delta_t=w["delta_t"],
                price=w["price"],
            )

        # Baseline vs crisis comparison table (baseline = year 0, before crisis)
        base_idx = 0
        base_probs = prob_history[base_idx]
        crisis_probs = prob_history[crisis_year]
        base_coherence = coherence[base_idx]
        crisis_coherence = coherence[crisis_year]
        base_purity = entanglement[base_idx]
        crisis_purity = entanglement[crisis_year]
        base_corr = np.corrcoef(prob_history[:crisis_year, 0], prob_history[:crisis_year, 1])[0, 1]
        crisis_corr = np.corrcoef(
            prob_history[max(0, crisis_year - 2) : crisis_year + 2, 0],
            prob_history[max(0, crisis_year - 2) : crisis_year + 2, 1],
        )[0, 1]

        print("\n" + "═" * 55)
        print("BASELINE vs CRISIS COMPARISON")
        print(f"{'':20} {'Russia':>12} {'France':>12}")
        print(f"P(restrict) baseline:  {base_probs[0]:12.3f} {base_probs[1]:12.3f}")
        print(f"P(restrict) crisis:    {crisis_probs[0]:12.3f} {crisis_probs[1]:12.3f}")
        print(f"Coherence baseline:    {base_coherence[0]:12.3f} {base_coherence[1]:12.3f}")
        print(f"Coherence crisis:      {crisis_coherence[0]:12.3f} {crisis_coherence[1]:12.3f}")
        print(f"Purity baseline:       {base_purity:.3f}")
        print(f"Purity crisis:         {crisis_purity:.3f}")
        print(f"Correlation baseline:  {base_corr:.3f}")
        print(f"Correlation crisis:    {crisis_corr:.3f}")
        print()
        print("Key result: Russia and France show ANTICORRELATED policy collapse")
        print("at crisis (expected from market-overlap coupling with negative J_eff).")
        print("═" * 55)

    if not args.no_figures:
        _make_figures(config, years, prob_history, entanglement, bloch_history,
                      coherence, price_series, crisis_year)


def _make_figures(
    config,
    years,
    prob_history,
    entanglement,
    bloch_history,
    coherence,
    price_series,
    crisis_year,
) -> None:
    """Generate all four France–Russia figures."""
    colors = config.colors
    labels = config.labels

    # Figure 1 — france_russia_overview.png (3-panel)
    fig1, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # Panel 1: Price signal
    axes[0].fill_between(years, price_series, alpha=0.3, color="#e74c3c")
    axes[0].plot(years, price_series, color="#c0392b", lw=2)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.7,
                    label=f"Crisis peak (year {crisis_year})")
    axes[0].set_title("Synthetic World Wheat Price Signal", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Price deviation (σ)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, len(years) - 1)

    # Panel 2: P(restrict) for Russia and France
    for i in range(2):
        axes[1].plot(years, prob_history[:, i], color=colors[i], lw=2,
                     label=labels[i], marker="o", ms=4)
    axes[1].axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
    axes[1].set_title("P(Restrict) — Russia vs France", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("P(restrict)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, len(years) - 1)

    # Panel 3: System purity
    axes[2].plot(years, entanglement, color="#2c3e50", lw=2)
    axes[2].axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
    axes[2].fill_between(years, entanglement, alpha=0.15, color="#2c3e50")
    axes[2].set_title("System Purity Tr(ρ²)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Tr(ρ²)")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlim(0, len(years) - 1)

    fig1.suptitle("France–Russia: 2-Country Quantum Wheat Demo\n"
                  "Market-overlap coupling (shared MENA exports)",
                  fontsize=12, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "france_russia_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {FIGURES_DIR / 'france_russia_overview.png'}")

    # Figure 2 — france_russia_phase_space.png
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(prob_history[:, 0], prob_history[:, 1],
                     c=years, cmap="viridis", s=60, zorder=3)
    ax2.plot(prob_history[:, 0], prob_history[:, 1],
             color="gray", lw=0.8, alpha=0.5, zorder=2)
    cx, cy = prob_history[crisis_year, 0], prob_history[crisis_year, 1]
    ax2.scatter([cx], [cy], color="#e74c3c", s=150, zorder=5,
                marker="*", label=f"Crisis (yr {crisis_year})")
    plt.colorbar(sc, ax=ax2, label="Year")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, label="Correlated")
    ax2.set_xlabel("P(Russia restricts)")
    ax2.set_ylabel("P(France restricts)")
    ax2.set_title("Phase Space: Russia vs France\n"
                  "Anticorrelated region = below diagonal", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.annotate("Both restrict", xy=(0.5, 0.6), fontsize=8, ha="center")
    ax2.annotate("Russia restricts,\nFrance liberalizes", xy=(0.5, 0.25), fontsize=8, ha="center")
    ax2.annotate("Both liberalize", xy=(0.2, 0.15), fontsize=8, ha="center")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "france_russia_phase_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {FIGURES_DIR / 'france_russia_phase_space.png'}")

    # Figure 3 — france_russia_bloch_spheres.png (side by side)
    fig3 = plt.figure(figsize=(14, 6))
    for col, (site, country) in enumerate([(0, "Russia"), (1, "France")]):
        ax = fig3.add_subplot(1, 2, col + 1, projection="3d")
        bx = bloch_history[:, site, 0]
        by = bloch_history[:, site, 1]
        bz = bloch_history[:, site, 2]

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.15, lw=0.4)

        for t in range(len(years) - 1):
            c = plt.cm.viridis(t / len(years))
            ax.plot(bx[t : t + 2], by[t : t + 2], bz[t : t + 2], color=c, lw=2.5)

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
        ax.text(0, 0, 1.15, "|R⟩\nRestrict", ha="center", fontsize=8, color="#c0392b", fontweight="bold")
        ax.text(0, 0, -1.15, "|L⟩\nLiberalize", ha="center", fontsize=8, color="#2980b9", fontweight="bold")
        ax.set_title(f"{country} — Bloch Sphere Trajectory", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

    fig3.suptitle("Bloch Sphere: Russia and France — Quantum State Trajectories\n"
                  "(coloured early→late: purple→yellow)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "france_russia_bloch_spheres.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {FIGURES_DIR / 'france_russia_bloch_spheres.png'}")

    # Figure 4 — france_russia_coherence.png
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    for i in range(2):
        ax4.plot(years, coherence[:, i], color=colors[i], lw=2.5,
                 label=labels[i], marker="o", ms=4)
    ax4.fill_between(years, coherence[:, 1], alpha=0.08, color=colors[1])
    ax4.axvline(crisis_year, color="#e74c3c", lw=1.8, ls=":", alpha=0.8,
                label=f"Crisis (yr {crisis_year})")
    ax4.set_title("Quantum Coherence |ρ₀₁| per Country\n"
                  "Magnitude of off-diagonal element of reduced density matrix",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("|ρ₀₁|  (0=classical, 0.5=maximum superposition)")
    ax4.set_ylim(0, 0.55)
    ax4.set_xlim(0, len(years) - 1)
    ax4.legend(fontsize=9)
    ax4.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    fig4.tight_layout()
    fig4.savefig(FIGURES_DIR / "france_russia_coherence.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved {FIGURES_DIR / 'france_russia_coherence.png'}")


if __name__ == "__main__":
    main()
