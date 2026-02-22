# Quantum Wheat Trade Network

A quantum decision theory model of the global wheat trade network, implementing an alternating Schrödinger / Lindblad simulation to capture simultaneous export restriction cascades during food price crises.

---

## Overview

### The Problem: Why Classical Cascade Models Fail

During the 2010–11 wheat crisis, multiple countries imposed export restrictions almost simultaneously—Russia, Ukraine, Kazakhstan, Argentina, and others—despite having no direct coordination. Classical cascade models (e.g., threshold contagion, game-theoretic best-response) struggle to explain this phenomenon because they assume:

1. **Sequential decision-making**: Country A restricts, then B observes and responds, then C, etc. In reality, announcements clustered within weeks.
2. **Independent information sets**: Each country is treated as reacting to its own local signal. Yet the *correlation* of restriction decisions across countries exceeded what local information alone would predict.
3. **Deterministic or weakly stochastic dynamics**: Small perturbations in initial conditions do not propagate into large, correlated policy shifts in the way observed.

The 2010–11 cascade exhibited **quantum-like** features: countries appeared to be in a *superposition* of “restrict” and “liberalize” until policy announcements *collapsed* the joint state. The build-up of correlations before collapse, and the sensitivity to global price signals, suggest a coherent quantum decision process rather than a classical sequential one.

### What Quantum Decision Theory Adds

Quantum decision theory (QDT) models each country’s policy choice as a two-level system: |L⟩ (liberalize) and |R⟩ (restrict). The key insight is that countries do not “have” a definite policy until they announce it—they exist in a superposition. The Hamiltonian couples countries through trade dependencies, so the joint state evolves coherently. Lindblad operators model the decoherence induced by policy announcements: when one country announces, it partially collapses the wavefunction of its trade partners.

This framework naturally produces:

- **Simultaneous cascades**: Correlated superpositions build up via Hamiltonian coupling; when decoherence occurs, multiple countries “collapse” in a correlated way.
- **Price sensitivity**: Global price spikes enter as time-dependent detuning, biasing import-dependent countries toward |R⟩.
- **Entanglement and coherence**: The purity Tr(ρ²) and off-diagonal elements of reduced density matrices quantify how “quantum” the system remains before announcements.

---

## Theoretical Framework

### 1. The N-Country Hamiltonian

$$
\hat{H} = \sum_i \delta_i \sigma_i^z + \sum_i \Delta_i \sigma_i^x + \sum_{i \lt j} J_{ij} \sigma_i^x \sigma_j^x
$$

**Mathematical explanation**: The Hamiltonian acts on an N-qubit Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes N}$. Each country $i$ is a qubit with basis states $|0\rangle \equiv |L\rangle$ (liberalize) and $|1\rangle \equiv |R\rangle$ (restrict). The Pauli operators $\sigma_i^z$ and $\sigma_i^x$ act on qubit $i$ only. The first term is the *detuning*: $\delta_i > 0$ biases toward $|R\rangle$, $\delta_i < 0$ toward $|L\rangle$. The second term is *tunneling*: $\Delta_i$ allows coherent transitions between $|L\rangle$ and $|R\rangle$ (policy flexibility). The third term is *Ising-type coupling*: $J_{ij} \sigma_i^x \sigma_j^x$ creates correlations between countries $i$ and $j$—when one flips, the other is nudged to flip as well. $J_{ij}$ is derived from bilateral trade shares (import dependency).

**Layman's explanation**: Think of each country as a compass needle that can point “liberalize” or “restrict.” The detuning $\delta_i$ is the domestic pressure: import-dependent countries feel a pull toward restriction when prices spike. The tunneling $\Delta_i$ is how easily a country can change its mind before committing. The coupling $J_{ij}$ is the trade link: if Russia restricts, Egypt (which imports 60% from Russia) is more likely to restrict too, because their “needles” are connected.

---

### 2. Schrödinger Equation and Unitary Evolution

$$
|\Psi(t)\rangle = e^{-i\hat{H}t/\hbar} |\Psi(0)\rangle
$$

(We set $\hbar = 1$.)

**Mathematical explanation**: For a closed quantum system, the state evolves unitarily: $|\Psi(t)\rangle = \hat{U}(t) |\Psi(0)\rangle$ with $\hat{U}(t) = e^{-i\hat{H}t}$. Unitarity preserves the norm and all superposition structure. The density matrix evolves as $\rho(t) = \hat{U}(t) \rho(0) \hat{U}^\dagger(t)$. During this phase, correlations between countries build up through the coupling terms in $\hat{H}$; no information is lost to the environment.

**Layman's explanation**: Between policy announcements, countries are “thinking” and influencing each other through trade links. No one has committed yet—the joint state is a coherent superposition of all possible policy combinations. This phase models the build-up of correlated uncertainty before governments announce their decisions.

---

### 3. Lindblad Master Equation

$$
\frac{d\rho}{dt} = -i[\hat{H}, \rho] + \sum_k \gamma_k \left( \hat{L}_k \rho \hat{L}_k^\dagger - \frac{1}{2}\{\hat{L}_k^\dagger \hat{L}_k, \rho\} \right)
$$

**Mathematical explanation**: The Lindblad equation is the most general form of a Markovian, trace-preserving, completely positive quantum master equation. The first term $-i[\hat{H}, \rho]$ is the unitary (von Neumann) evolution. The second term describes decoherence: each Lindblad operator $\hat{L}_k$ models a channel through which the system couples to the environment. In our implementation, we use $\hat{L}_k = \sqrt{\gamma_k} \sigma_k^z$ (dephasing), which suppresses off-diagonal coherence of qubit $k$ at rate $\gamma_k$. The anticommutator $\{\hat{L}_k^\dagger \hat{L}_k, \rho\}$ ensures trace preservation.

**Layman's explanation**: When a country announces its policy (or when news, rumors, and market signals accumulate), the “quantum fog” of uncertainty collapses. The Lindblad term models this: $\gamma_k$ is how often country $k$’s policy gets “measured” by the environment (announcements, press releases). High $\gamma$ means rapid decoherence—the country quickly commits to a definite policy.

---

### 4. Partial Trace / Reduced Density Matrix

$$
\rho_i = \mathrm{Tr}_{j \neq i}[\rho]
$$

**Mathematical explanation**: The full density matrix $\rho$ lives on $\mathcal{H}^{\otimes N}$. To obtain the state of country $i$ alone, we trace out all other degrees of freedom: $\rho_i = \mathrm{Tr}_{j \neq i}[\rho]$. This yields a $2 \times 2$ density matrix for qubit $i$. If $\rho$ is entangled, $\rho_i$ is mixed ($\mathrm{Tr}(\rho_i^2) < 1$); if $\rho$ is separable, $\rho_i$ can be pure.

**Layman's explanation**: The reduced density matrix $\rho_i$ answers: “If we ignore all other countries and look only at country $i$, what is its effective state?” It captures both the local bias (diagonal elements) and how much that country is still in a superposition (off-diagonal elements), given its entanglement with the rest of the network.

---

### 5. Bloch Vector

$$
\vec{b}_i = (\langle \sigma^x \rangle_i, \langle \sigma^y \rangle_i, \langle \sigma^z \rangle_i) = (\mathrm{Tr}[\sigma^x \rho_i], \mathrm{Tr}[\sigma^y \rho_i], \mathrm{Tr}[\sigma^z \rho_i])
$$

**Mathematical explanation**: Any single-qubit density matrix can be written $\rho_i = \frac{1}{2}(I + \vec{b}_i \cdot \vec{\sigma})$, where $\vec{\sigma} = (\sigma^x, \sigma^y, \sigma^z)$. The Bloch vector $\vec{b}_i$ lies in or on the unit ball: $|\vec{b}_i| \leq 1$. Pure states lie on the surface; mixed states lie inside. The $z$-component $\langle \sigma^z \rangle$ is the population difference: $+1$ means fully $|R\rangle$, $-1$ means fully $|L\rangle$. The $x$ and $y$ components encode coherence (phase between $|L\rangle$ and $|R\rangle$).

**Layman's explanation**: The Bloch vector is a 3D arrow representing where country $i$ “points” in policy space. North pole = restrict, south pole = liberalize. The length of the arrow indicates how definite the policy is: a short arrow means the country is still in a superposition. The $x$ and $y$ components capture the “quantum wobble” before commitment.

---

### 6. Restriction Probability

$$
P(R)_i = \mathrm{Tr}[\Pi_i^R \rho] = \mathrm{Tr}[\rho \Pi_i^R]
$$

where $\Pi_i^R = |1\rangle\langle 1|_i$ projects onto the restrict state at site $i$.

**Mathematical explanation**: The probability that country $i$ is observed in state $|R\rangle$ is given by the Born rule: $P(R)_i = \langle 1 | \rho_i | 1 \rangle = (\rho_i)_{11}$. Equivalently, $P(R)_i = \mathrm{Tr}[\Pi_i^R \rho]$ where $\Pi_i^R$ is the projector onto $|R\rangle$ at site $i$ (and identity elsewhere). This is the observable we track over time.

**Layman's explanation**: $P(R)_i$ is the probability that country $i$ will restrict exports if we “measure” it now. It’s the main output of the model: a time series of restriction probabilities for each country, driven by price and coupling.

---

### 7. Quantum Coherence

$$
|\rho_{01}| = |(\rho_i)_{01}|
$$

**Mathematical explanation**: For a single-qubit density matrix $\rho_i$, the off-diagonal element $(\rho_i)_{01} = \langle 0 | \rho_i | 1 \rangle$ quantifies the coherence between $|L\rangle$ and $|R\rangle$. For a pure superposition $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, $|\rho_{01}| = |\alpha\beta^*|$, maximized at $1/2$ when $|\alpha| = |\beta| = 1/\sqrt{2}$. Decoherence drives $|\rho_{01}| \to 0$; $|\rho_{01}| = 0$ means the state is classical (a mixture of $|L\rangle$ and $|R\rangle$ with no phase relationship).

**Layman's explanation**: $|\rho_{01}|$ measures how much a country is “in two minds.” Zero means it has effectively chosen (or been forced to choose). A value near 0.5 means it is in maximum superposition—truly undecided. During a crisis, we expect coherence to spike as countries hesitate, then drop as announcements roll in.

---

## Parameter Guide

| Parameter | Physical meaning | Empirical source (real model) | Synthetic value (this repo) |
|-----------|------------------|-------------------------------|-----------------------------|
| $\delta_i$ (detuning) | Domestic bias toward restriction vs. liberalization | World Bank WGI (governance, political stability), self-sufficiency ratio | $\delta_{\mathrm{base}}$: Russia −0.3, USA −0.4, Egypt +0.2, Tunisia +0.3, Lebanon +0.4; time-dependent via $\delta_i(t) = \delta_{\mathrm{base},i} + w_i \cdot p(t)$ |
| $w_i$ (price sensitivity) | How much a price spike pushes country $i$ toward restriction | Import dependency, dietary share of wheat | Exporters: 0.2; Importers: 0.9 (Egypt), 1.1 (Tunisia), 1.3 (Lebanon) |
| $\Delta_i$ (tunneling) | Policy flexibility / ease of switching | Institutional rigidity, policy bandwidth | 0.5 (Russia), 0.6 (USA), 0.3 (Egypt), 0.2 (Tunisia), 0.15 (Lebanon) |
| $J_{ij}$ (coupling) | Strength of policy correlation between $i$ and $j$ | FAO/FAOSTAT bilateral trade matrix (share of $i$’s imports from $j$) | Egypt: 60% Russia, 25% USA; Tunisia: 50% Russia, 30% USA; Lebanon: 70% Russia, 15% USA |
| $\gamma_i$ (decoherence rate) | Frequency of policy announcements / information shocks | Media coverage, transparency indices | 0.15 (Russia), 0.12 (USA), 0.20 (Egypt), 0.25 (Tunisia), 0.30 (Lebanon) |
| $p(t)$ (price signal) | World wheat price deviation from trend | TWIST model output, FAO price indices | Synthetic: baseline cycle + Gaussian crisis spike at year 10 |

---

## Simulation Architecture

Each annual timestep follows an alternating Schrödinger / Lindblad loop:

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    INITIAL STATE                         │
                    │              |+⟩^⊗N  (equal superposition)               │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              SCHRÖDINGER PHASE (dt ≈ 0.7 yr)             │
                    │  • Compute H from price p(t) and δ(t) = δ_base + w·p(t)  │
                    │  • Unitary evolution: ρ → U ρ U†                         │
                    │  • Coherent build-up of correlations via J_ij σᵢˣσⱼˣ   │
                    │  • No decoherence — pure quantum evolution                │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │               LINDBLAD PHASE (dt ≈ 0.3 yr)               │
                    │  • Same H; add Lindblad operators Lₖ = √γₖ σₖᶻ          │
                    │  • Integrate dρ/dt = -i[H,ρ] + Σₖ(LₖρLₖ† - ½{Lₖ†Lₖ,ρ})   │
                    │  • Decoherence: policy announcements collapse superpos.  │
                    │  • Purity Tr(ρ²) decreases; off-diagonals decay          │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                    OBSERVABLES                           │
                    │  • P(R)_i = Tr[Πᵢᴿ ρ]  for each country i                │
                    │  • Tr(ρ²)  (purity)                                       │
                    │  • Reduced ρᵢ, Bloch vectors, |ρ₀₁| (coherence)          │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                   NEXT TIMESTEP                          │
                    │  • Advance t → t+1; update price p(t+1)                   │
                    │  • Use ρ(t) as initial condition for next year          │
                    │  • Repeat until t = T_YEARS                              │
                    └─────────────────────────────────────────────────────────┘
```

---

## Output Figures

The simulation produces **5 figures** (1 from the core simulation, 4 from the wavefunction visualisation):

### 1. `quantum_wheat_synthetic.png` (core simulation)

A 6-panel figure:

- **Panel 6a**: Synthetic world wheat price signal over 20 years, with a crisis spike at year 10. Shows the exogenous driver of the model.
- **Panel 6b**: Restriction probabilities for exporters (Russia, USA) over time. Expect lower baseline P(R) and a rise during the crisis.
- **Panel 6c**: Restriction probabilities for importers (Egypt, Tunisia, Lebanon). Expect higher P(R) and stronger crisis response.
- **Panel 6d**: System purity Tr(ρ²). Values near 1 indicate a nearly pure (coherent) state; values below 1 indicate decoherence. Watch for dips during the Lindblad phase and recovery during the Schrödinger phase.
- **Panel 6e**: Phase space (Russia vs. Egypt). Each point is a year; colour indicates time. The trajectory shows how the two countries’ restriction probabilities co-evolve—look for clustering near the crisis.
- **Panel 6f**: Policy correlation matrix: baseline (years 0–9) vs. post-crisis (years 10–19). Red = correlated, blue = anticorrelated. Expect increased correlation among importers and between Russia and its import partners after the crisis.

### 2. `wavefunction_bloch_components.png`

Time series of the three Bloch vector components ⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩ for each country. ⟨σᶻ⟩ is the restriction bias (+1 = restrict, −1 = liberalize). ⟨σˣ⟩ and ⟨σʸ⟩ encode coherence and phase. Look for oscillations and crisis-induced shifts.

### 3. `wavefunction_basis_distribution.png`

Bar charts of the probability distribution over the 32 basis states |b₀b₁b₂b₃b₄⟩ at key moments (t=0, 5, 10, 11, 19). Each bar is |⟨b|Ψ⟩|². The rightmost bar (11111) is the all-restrict state. Watch how probability mass shifts toward high-restriction states during the crisis.

### 4. `wavefunction_coherence.png`

Magnitude of the off-diagonal element |ρ₀₁| for each country over time. 0 = classical; 0.5 = maximum superposition. Expect coherence to vary with the Schrödinger/Lindblad cycle and to be sensitive to the crisis.

### 5. `wavefunction_bloch_sphere.png`

3D Bloch sphere trajectories for Russia and Egypt. The state vector traces a path on the unit sphere; colour indicates time (early→late). North pole = restrict, south pole = liberalize. Green = t=0, red star = crisis year, purple = final year. Visualises how each country’s quantum state moves through policy space.

---

## Roadmap

This repository implements **synthetic validation** of the quantum wheat trade model. Parameters (J, δ, Δ, γ) and the price series are chosen for illustration, not calibration.

**Next steps**:

1. **Replace J matrix** with real FAO/FAOSTAT bilateral wheat trade data (import shares by country pair).
2. **Replace price series** with TWIST model output or FAO price indices for historical validation (e.g., 2010–11, 2022).
3. **Calibrate δ, Δ, γ** using World Bank WGI, institutional indices, and observed announcement frequencies.

**Reference**: Kuhla, Puma & Otto (2024), *Communications Earth & Environment* — for the TWIST model and trade network context.

---

## Citation / Acknowledgements

*Placeholder for paper citation once submitted.*

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run core simulation (produces quantum_wheat_synthetic.png)
python src/quantum_wheat_synthetic.py

# Run wavefunction visualisation (produces 4 additional figures)
python src/quantum_wheat_wavefunctions.py
```

Figures are saved to the `figures/` directory.
