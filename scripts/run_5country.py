"""
Quantum Wheat Trade Network — 5-Country Synthetic Demo (ported)
================================================================

Russia, USA (exporters) + Egypt, Tunisia, Lebanon (importers).

Coupling mechanism: IMPORT DEPENDENCY.
J[i,j] = fraction of country i's wheat imports sourced from country j.
This is a supply-chain coupling: importers' policy states are entangled
with exporters' through trade dependency. Contrast with the France–Russia
2-country model, which uses MARKET OVERLAP (shared export destinations).

Uses quantum_wheat.core and make_5country_config(). Output figures must be
numerically identical to the original quantum_wheat_synthetic.py given seed 42.
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from quantum_wheat import run_simulation, make_5country_config

warnings.filterwarnings("ignore")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = make_5country_config()
    rng = np.random.default_rng(42)

    result = run_simulation(config, rng)

    prob_history = result["prob_history"]
    entanglement = result["entanglement"]
    price_series = result["price_series"]

    N = config.n
    T_YEARS = config.T_years
    LABELS = config.labels
    COLORS = config.colors
    crisis_year = config.crisis_year
    years = np.arange(T_YEARS)

    print("Coupling matrix J (import dependency):")
    print(np.round(config.J, 2))
    print()
    print("Running simulation...")
    print("Simulation complete.\n")

    # Build figure — identical layout to original quantum_wheat_synthetic.py
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # a. Price signal
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

    # b. Exporters: P(restrict)
    ax1 = fig.add_subplot(gs[1, 0])
    for i in [0, 1]:
        ax1.plot(years, prob_history[:, i], color=COLORS[i],
                 lw=2, label=LABELS[i], marker="o", ms=4)
    ax1.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
    ax1.set_title("Exporters: P(Restrict)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("P(restrict)")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, T_YEARS - 1)

    # 6c. Importers: P(restrict)
    ax2 = fig.add_subplot(gs[1, 1])
    for i in [2, 3, 4]:
        ax2.plot(years, prob_history[:, i], color=COLORS[i],
                 lw=2, label=LABELS[i], marker="o", ms=4)
    ax2.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
    ax2.set_title("Importers: P(Restrict / Hoard)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("P(restrict)")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, T_YEARS - 1)

    # d. System purity
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(years, entanglement, color="#2c3e50", lw=2)
    ax3.axvline(crisis_year, color="#e74c3c", lw=1.5, ls=":", alpha=0.6)
    ax3.fill_between(years, entanglement, alpha=0.15, color="#2c3e50")
    ax3.set_title("System Purity  Tr(ρ²)\n(1 = pure, <1 = decoherence / classical)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Tr(ρ²)")
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(0, T_YEARS - 1)

    # e. Phase space: Russia vs Egypt
    ax4 = fig.add_subplot(gs[2, 1])
    sc = ax4.scatter(prob_history[:, 0], prob_history[:, 2],
                     c=years, cmap="viridis", s=60, zorder=3)
    ax4.plot(prob_history[:, 0], prob_history[:, 2],
             color="gray", lw=0.8, alpha=0.5, zorder=2)
    cx, cy = prob_history[crisis_year, 0], prob_history[crisis_year, 2]
    ax4.scatter([cx], [cy], color="#e74c3c", s=150, zorder=5,
                marker="*", label=f"Crisis (yr {crisis_year})")
    plt.colorbar(sc, ax=ax4, label="Year")
    ax4.set_xlabel("P(Russia restricts)")
    ax4.set_ylabel("P(Egypt restricts)")
    ax4.set_title("Phase Space: Russia vs Egypt", fontsize=11, fontweight="bold")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=9)

    # f. Correlation heatmap
    ax5 = fig.add_subplot(gs[3, :])
    baseline = prob_history[:crisis_year, :]
    post = prob_history[crisis_year:, :]

    corr_base = np.corrcoef(baseline.T)
    corr_post = np.corrcoef(post.T)

    combined = np.hstack([corr_base, np.full((N, 1), np.nan), corr_post])
    im = ax5.imshow(combined, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax5, fraction=0.02)

    ticks = list(range(N)) + [N] + list(range(N + 1, 2 * N + 1))
    tick_labels = LABELS + [""] + LABELS
    ax5.set_xticks(ticks)
    ax5.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax5.set_yticks(range(N))
    ax5.set_yticklabels(LABELS, fontsize=8)
    ax5.set_title(
        "Policy Correlation Matrix:  Baseline (years 0–9, left)  vs  Post-Crisis (years 10–19, right)\n"
        "Blue = anticorrelated  |  Red = correlated",
        fontsize=11, fontweight="bold"
    )
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
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Print diagnostics
    print("\n── Crisis response (year 10) ──────────────────────────────────────")
    for i in range(N):
        pre = prob_history[crisis_year - 1, i]
        peak = prob_history[crisis_year, i]
        print(f"  {LABELS[i]:10s}  pre={pre:.3f}  peak={peak:.3f}  Δ={peak-pre:+.3f}")

    print("\n── Correlation shift (baseline vs post-crisis) ──────────────────")
    pairs = [(0, 2, "Russia–Egypt"), (0, 3, "Russia–Tunisia"),
             (0, 4, "Russia–Lebanon"), (2, 3, "Egypt–Tunisia")]
    for i, j, name in pairs:
        b = np.corrcoef(baseline[:, i], baseline[:, j])[0, 1]
        p = np.corrcoef(post[:, i], post[:, j])[0, 1]
        print(f"  {name:20s}  baseline={b:+.3f}  post-crisis={p:+.3f}  Δ={p-b:+.3f}")


if __name__ == "__main__":
    main()
