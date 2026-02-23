# Figure Captions — Student Guide

**How to use this guide:** Run the simulation scripts, then open each figure alongside this document. Work through the captions in order; they explain what each panel shows and how to interpret it.

---

## Figure 1: `quantum_wheat_synthetic.png`

**Overall:** This is the main output of the core simulation. It shows how the 5-country quantum system responds to a synthetic wheat price crisis over 20 years.

### Panel a — Price Signal (top, full width)

**What it shows:** The exogenous driver of the model: world wheat price deviation from its long-run mean. Values above zero mean prices are higher than average.

**How to read it:**
- The **dashed vertical line** marks year 10, when the crisis peaks.
- The **red shaded area** is the price deviation; the spike around year 10 is a Gaussian “crisis” added to a baseline ~6-year commodity cycle.
- Small random noise is added each year (reproducible with seed 42).

**What to look for:** The crisis is artificial but realistic in shape—a sharp spike that decays. This is the input that pushes import-dependent countries toward restriction via the detuning δ(t) = δ_base + w·p(t).

---

### Panel b — Exporters: P(Restrict) (middle left)

**What it shows:** The probability that Russia (red) and USA (blue) will restrict wheat exports in each year.

**How to read it:**
- **Y-axis:** P(restrict) from 0 to 1. At 0.5, the country is equally likely to restrict or liberalize.
- **X-axis:** Year (0–19).
- **Dashed line:** Crisis year 10.

**What to look for:** Exporters have negative baseline detuning (δ_base), so they are biased toward liberalize. During the crisis, the price spike increases δ(t), so P(restrict) may rise slightly. Compare the two exporters: Russia (heavily imported by Egypt, Tunisia, Lebanon) may respond differently from the USA due to coupling J.

---

### Panel c — Importers: P(Restrict / Hoard) (middle right)

**What it shows:** The probability that Egypt (green), Tunisia (orange), and Lebanon (purple) will restrict imports or hoard in each year.

**How to read it:** Same axes as Panel b. Importers have positive δ_base and higher price sensitivity (w), so they are more vulnerable to price spikes.

**What to look for:** Importers should generally have higher P(restrict) than exporters in baseline years, and a stronger response to the crisis. Lebanon (most vulnerable, w=1.3) may show the largest crisis response. Compare the three importers: their correlation with Russia (via J) affects how they move together.

---

### Panel d — System Purity Tr(ρ²) (bottom left)

**What it shows:** The purity of the full N-qubit density matrix. Tr(ρ²) = 1 for a pure state (fully coherent); Tr(ρ²) < 1 for a mixed state (decoherence has occurred).

**How to read it:**
- **Y-axis:** Purity from 0 to 1.
- **Value 1:** The joint state is a pure quantum superposition.
- **Value < 1:** Decoherence has reduced coherence; the system is more “classical.”

**What to look for:** Purity typically decreases over time as Lindblad decoherence accumulates. It may dip around the crisis year when the Hamiltonian and decoherence interact strongly. The alternating Schrödinger (coherent) and Lindblad (decoherent) phases create a step-like or oscillatory pattern.

---

### Panel e — Phase Space: Russia vs Egypt (bottom right)

**What it shows:** A 2D phase portrait of how Russia’s and Egypt’s restriction probabilities co-evolve. Each point is one year; colour indicates time (purple = early, yellow = late).

**How to read it:**
- **X-axis:** P(Russia restricts).
- **Y-axis:** P(Egypt restricts).
- **Red star:** Crisis year (year 10).
- **Gray line:** Trajectory connecting years in order.

**What to look for:** The trajectory shows whether Russia and Egypt move together (positive slope) or in opposition (negative slope). A cluster of points near the crisis suggests both countries respond similarly in that period. The correlation between Russia and Egypt (via J: Egypt imports 60% from Russia) should be visible in how the trajectory bends or clusters.

---

### Panel f — Policy Correlation Matrix (bottom, full width)

**What it shows:** Two 5×5 correlation matrices side by side. **Left:** Correlation of P(restrict) across countries in baseline years (0–9). **Right:** Same for post-crisis years (10–19). Each cell is the Pearson correlation between one country’s P(restrict) time series and another’s.

**How to read it:**
- **Rows and columns:** Russia, USA, Egypt, Tunisia, Lebanon (same order in both matrices).
- **Red:** Positive correlation (when one restricts, the other tends to restrict).
- **Blue:** Negative correlation (when one restricts, the other tends to liberalize).
- **White:** Near zero (no linear relationship).

**What to look for:** Compare left vs. right. The key result is that **Russia–Egypt, Russia–Tunisia, Russia–Lebanon** correlations should increase (become more red) after the crisis. This reflects the build-up of policy entanglement via the Hamiltonian coupling J during the coherent phase, and correlated collapse during decoherence.

---

## Figure 2: `wavefunction_bloch_components.png`

**Overall:** Three stacked time-series plots showing the Bloch vector components (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩) for each country’s reduced quantum state. This reveals the “quantum” structure of each country’s decision state.

### Top panel — ⟨σˣ⟩ (quantum coherence)

**What it shows:** The x-component of the Bloch vector. Non-zero ⟨σˣ⟩ means the country is in a superposition of |L⟩ and |R⟩ with a definite phase relationship.

**How to read it:** ⟨σˣ⟩ ∈ [−1, 1]. Zero means no coherence in the x-basis; ±1 means maximum coherence. The dashed line at 0 is the reference.

**What to look for:** Coherence tends to decrease over time as decoherence acts. Different countries (different colours) may have different coherence levels depending on their coupling and decoherence rate γ.

---

### Middle panel — ⟨σʸ⟩ (phase)

**What it shows:** The y-component of the Bloch vector. Encodes the phase between |L⟩ and |R⟩ in the superposition.

**How to read it:** ⟨σʸ⟩ ∈ [−1, 1]. Zero means the phase is 0 or π; non-zero means a non-trivial phase.

**What to look for:** ⟨σʸ⟩ can oscillate as the Hamiltonian drives coherent evolution. It is sensitive to the relative phase built up during the Schrödinger phase.

---

### Bottom panel — ⟨σᶻ⟩ (restriction bias)

**What it shows:** The z-component of the Bloch vector. ⟨σᶻ⟩ = +1 means fully |R⟩ (restrict); ⟨σᶻ⟩ = −1 means fully |L⟩ (liberalize).

**How to read it:** ⟨σᶻ⟩ ∈ [−1, 1]. This is the most intuitive: it directly maps to “how much” the country is leaning toward restriction vs. liberalization.

**What to look for:** Exporters (Russia, USA) should have negative or small ⟨σᶻ⟩ in baseline; importers (Egypt, Tunisia, Lebanon) should have positive ⟨σᶻ⟩. During the crisis, ⟨σᶻ⟩ may shift toward +1 (restrict) for importers as the price spike increases their detuning.

---

## Figure 3: `wavefunction_basis_distribution.png`

**Overall:** Five bar charts showing the probability distribution over the 32 basis states |b₀b₁b₂b₃b₄⟩ at key moments: t=0 (initial), t=5 (mid-baseline), t=10 (crisis, post-Schrödinger), t=11 (post-crisis), t=19 (final). Each bar is |⟨b|Ψ⟩|².

**Convention:** Bit 0 = Liberalize (L), Bit 1 = Restrict (R). Order: Russia, USA, Egypt, Tunisia, Lebanon. So the rightmost bar (index 31 = 11111) is the state where all five countries restrict.

**How to read it:**
- **X-axis:** Basis state index (0–31). Labels show binary strings; e.g. 11111 = all restrict.
- **Y-axis:** Probability (0–1). Bars sum to 1.
- **Red dashed line:** Marks the all-restrict state (index 31).

**What to look for:**
- **t=0:** Initial state is |+⟩^⊗N (equal superposition), so all bars have equal height (1/32 ≈ 0.031).
- **t=5:** Probability mass may have shifted toward states with more 1s (restrict) for importers.
- **t=10 (crisis):** Look for increased probability on high-restriction states (right side). The crisis should concentrate probability mass.
- **t=11, t=19:** Compare how the distribution evolves after the crisis. Decoherence narrows the distribution toward more classical (diagonal) states.

---

## Figure 4: `wavefunction_coherence.png`

**Overall:** Time series of the quantum coherence |ρ₀₁| for each country. This is the magnitude of the off-diagonal element of each country’s reduced density matrix—a direct measure of “how much in superposition” each country is.

**What it shows:** |ρ₀₁| ∈ [0, 0.5]. Zero means the state is classical (a mixture of |L⟩ and |R⟩ with no phase). 0.5 is the maximum for a single qubit (equal superposition |+⟩).

**How to read it:**
- **Y-axis:** |ρ₀₁| from 0 to 0.55. The dashed line at 0.5 marks maximum superposition.
- **Dashed vertical line:** Crisis year 10.
- **Colours:** Same as other figures (Russia, USA, Egypt, Tunisia, Lebanon).

**What to look for:** Coherence generally decreases over time as Lindblad dephasing acts. Countries with higher γ (e.g. Lebanon, Tunisia) decohere faster. During the crisis, coherence may drop sharply as the system is driven toward more classical (committed) states. Compare baseline (years 0–9) to crisis and post-crisis.

---

## Figure 5: `wavefunction_bloch_sphere.png`

**Overall:** Two 3D Bloch spheres (one for Russia, one for Egypt) showing how each country’s reduced quantum state moves on the unit sphere over the 20-year simulation.

**What it shows:** The Bloch sphere is the space of single-qubit states. The north pole (z=+1) is |R⟩ (restrict); the south pole (z=−1) is |L⟩ (liberalize). The equator (z=0) corresponds to equal superposition. The trajectory is coloured by time (purple = early, yellow = late).

**How to read it:**
- **Wireframe sphere:** The unit sphere; all valid single-qubit states lie on or inside it.
- **Trajectory:** The path of the Bloch vector (⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩) over time.
- **Green dot:** t=0 (initial state).
- **Red star:** Crisis year (t=10).
- **Purple dot:** Final year (t=19).

**What to look for:**
- **Russia:** As an exporter with negative δ_base, the trajectory may start and stay in the southern hemisphere (liberalize bias). During the crisis, it may move northward.
- **Egypt:** As an importer with positive δ_base and strong coupling to Russia (J=0.6), the trajectory may be more in the northern hemisphere. Watch how Egypt’s path responds to the crisis.
- **Trajectory shape:** Smooth curves indicate coherent evolution; kinks or jumps may reflect decoherence events. The trajectory should remain on or inside the sphere (pure states on the surface, mixed inside).

---

## Quick Reference: Colour Legend

| Colour  | Country  |
|---------|----------|
| Red     | Russia   |
| Blue    | USA      |
| Green   | Egypt    |
| Orange  | Tunisia  |
| Purple  | Lebanon  |

---

## Suggested Reading Order

1. **quantum_wheat_synthetic.png** — Start here. Get the big picture: price → restriction probabilities → correlations.
2. **wavefunction_bloch_components.png** — Understand the quantum structure (Bloch vector) behind the probabilities.
3. **wavefunction_coherence.png** — See how coherence decays and responds to the crisis.
4. **wavefunction_basis_distribution.png** — See the full 32-dimensional state at key moments.
5. **wavefunction_bloch_sphere.png** — Visualise Russia and Egypt’s trajectories in 3D policy space.
