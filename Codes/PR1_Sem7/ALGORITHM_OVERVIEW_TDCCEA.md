# Trajectory-Driven Cooperative Coevolutionary Algorithms (TD-CCEA Family)

## Comprehensive Algorithmic Overview & Implementation Reference

This document provides a detailed breakdown of the **TDCCEA**, **SA-TDCCEA**, and **CCE-TDCCEA** algorithms based on their codebase implementations in [`SEM7_PR1_Results.py`](./SEM7_PR1_Results.py), [`CCE_TDCCEA.py`](../CCE_TDCCEA.py), and [`SA_TDCCEA.py`](../SA_TDCCEA.py).

---

## 1. Context & Strategy Representation

All three algorithms evolve continuous strategy vectors for **microRTS** (`EvolvedBrainAI`). Instead of selecting among discrete scripts, agents are parameterized by a **20-dimensional continuous weight vector** $\boldsymbol{\theta} \in [0, 1]^{20}$:

```text
Macro / Economy Sub-Genome (8 Dimensions):
  Indices: [0, 1, 2, 6, 14, 15, 17, 18]
  Parameters: [
    worker_ratio, max_workers, expand_threshold, barracks_timing,
    economy_pressure, scout_priority, second_barracks, worker_fight_ratio
  ]

Micro / Combat Sub-Genome (12 Dimensions):
  Indices: [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]
  Parameters: [
    light_preference, heavy_preference, ranged_preference, aggression,
    attack_threshold, defend_radius, target_priority, defense_trigger,
    counter_rush, tech_switch, group_attack, focus_fire
  ]
```

Evaluations take place via Python $\leftrightarrow$ Java IPC over subprocess pipes (`tests.EvolvedBatchGameRunner`), simulating games on dynamic $48\times 48$ maps featuring progressive BFS-validated obstacle placement.

---

## 2. TDCCEA (Trajectory-Driven Cooperative Coevolutionary Algorithm)

### 2.1 Core Problem Addressed
Standard Genetic Algorithms suffer from **catastrophic forgetting** and **premature convergence** because they evaluate individuals purely on single-generation (momentary) fitness. In dynamic and co-evolving environments, an agent's true quality depends on its performance across varying environmental states. 

**TD-CCEA** solves this by maintaining a **sliding-window trajectory memory** $(f_i^{(k)}, \mathbf{c}_k)$ of past evaluations, weighting past performances by the similarity of the environment meta-state, and adapting mutation step sizes dynamically.

### 2.2 Algorithmic Workflow
```
 ┌────────────────────────────────────────────────────────┐
 │ 1. Initialize Population P₀ of N=20 agents             │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Compute Population Centroid c_t (Environment State) │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Simulate K Games & Measure Raw Fitness f_i          │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Update Trajectory Memory H_i ← H_i ∪ {(f_i, c_t)}   │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 5. Compute Dynamic Similarity Weights α_k & Score e_i  │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 6. Calculate Adaptive Mutation Rate μ_i = μ_base(1-e_i)│
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 7. Diversity Pruning (δ_s, δ_f) & Elite Archive A_max  │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 8. Breed Next Generation (Tournament + Gaussian Noise) │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            └──► (Repeat for T_max Generations)
```

### 2.3 Step-by-Step Mathematical & Code Breakdown:
1. **Population Centroid as Environment Meta-State**:
   At generation $t$, the algorithm calculates the centroid $\mathbf{c}_t$ representing the average strategic position of the current population:
   $$\mathbf{c}_t = \frac{1}{N} \sum_{i=1}^N \boldsymbol{\theta}_i^t$$

2. **Simulated Evaluation**:
   Each agent $\mathcal{O}_i$ plays $K$ matches (either against rival algorithm elites or rotating bot archetypes) to compute instantaneous raw fitness $f_i^t \in [0, 1]$.

3. **Trajectory Memory Storage**:
   Each agent maintains a sliding history $H_i = \{(f_i^{(1)}, \mathbf{c}_1), \dots, (f_i^{(k)}, \mathbf{c}_k)\}$ (capped at window size $W = 7$):
   ```python
   def update_history(self, f, env):
       self.H_i.append((f, env))
       if len(self.H_i) > 7:
           self.H_i.pop(0)
   ```

4. **Dynamic Similarity Weighting & Trajectory Score**:
   Past fitness scores are weighted based on the Euclidean distance between past centroid $\mathbf{c}_k$ and current centroid $\mathbf{c}_t$:
   $$\alpha_k = \max\left(0,\, 1 - \frac{\|\mathbf{c}_t - \mathbf{c}_k\|_2}{\sqrt{D}}\right), \quad \hat{\alpha}_k = \frac{\alpha_k}{\sum_j \alpha_j}$$
   The comprehensive trajectory score $e_i^t$ is computed as:
   $$e_i^t = \sum_k \hat{\alpha}_k \cdot f_i^{(k)}$$

5. **Context-Aware Adaptive Mutation Rate**:
   Step sizes scale inversely with trajectory performance:
   $$\mu_i = \mu_{\text{base}} \cdot (1 - e_i^t) + \epsilon \quad (\mu_{\text{base}} = 0.15, \, \epsilon = 0.03)$$
   * **High performers ($e_i \to 1.0$)**: receive small $\mu_i \approx 0.03$ (fine-tuning near optima).
   * **Poor performers ($e_i \to 0.0$)**: receive large $\mu_i \approx 0.18$ (exploring distant parameter regions).

6. **Diversity Pruning & Elite Archive**:
   * **Clustering Pruning**: If an agent is too close to an already selected survivor ($1 - \frac{d}{\sqrt{D}} > \delta_s = 0.85$), it is discarded.
   * **Fitness Pruning**: Stagnant weak agents ($e_i < \delta_f = 0.20$ after $\ge 3$ evaluations) are culled.
   * **Archive $\mathcal{A}$ ($\le 15$)**: The top 3 performers of each generation are stored in an external archive $\mathcal{A}$ to preserve historical best strategies.

7. **Reproduction & Breeding**:
   Offspring are produced via 3-way tournament selection from $\text{Survivors} \cup \mathcal{A}$ with additive Gaussian noise:
   $$\boldsymbol{\theta}_{\text{child}} = \operatorname{clip}\left(\boldsymbol{\theta}_{\text{parent}} + \mathcal{N}(0, \mu_i^2),\, 0,\, 1\right)$$

---

## 3. SA-TDCCEA (Sensitivity-Adaptive TD-CCEA)

### 3.1 Core Problem Addressed
Standard TD-CCEA uses a single scalar mutation rate $\mu_i$ applied uniformly across all 20 dimensions. However, in RTS games, different parameters exhibit vastly different sensitivities:
* **High-sensitivity weights** (e.g., `aggression`, `attack_threshold`) will cause immediate tactical collapse if mutated randomly.
* **Low-sensitivity weights** (e.g., `second_barracks`, `scout_priority`) require aggressive exploratory shifts to produce observable differences.

**SA-TDCCEA** computes an **empirical parameter influence profile** and applies **per-weight mutation rates** $\mu_j$.

### 3.2 Step-by-Step Execution:

```
[ Phase 1: Influence Analysis ]
  Baseline θ = [0.5]^20
  For each gene j in 0..19:
    Probe w_j = 0.0 vs w_j = 1.0 → Measure Δ win rate → Influence I_j = |r_high - r_low|
  Map Influence to Inverted Mutation Rates:
    μ_j = 0.08 + (1.0 - I_j / I_max) * 0.12  ∈ [0.08, 0.20]

[ Phase 2: Dynamic Per-Gene Reproduction ]
  During Offspring Generation:
    σ_j = μ_j * (1.0 - e_i) + 0.02
    θ_child[j] = clip(θ_parent[j] + N(0, σ_j^2), 0.0, 1.0)
```

1. **Phase 1: Influence Profiling**:
   Before or at the beginning of evolution:
   * Evaluate $\boldsymbol{\theta}_{\text{low}}$ where $w_j = 0.0$ (others $0.5$) $\to r_{\text{low}}$.
   * Evaluate $\boldsymbol{\theta}_{\text{high}}$ where $w_j = 1.0$ (others $0.5$) $\to r_{\text{high}}$.
   * Influence score: $I_j = |r_{\text{high}} - r_{\text{low}}|$.

2. **Phase 2: Inverted Sensitivity Mutation Mapping**:
   $$\mu_j = 0.08 + \left(1.0 - \frac{I_j}{I_{\max}}\right) \times 0.12 \quad \in [0.08, \, 0.20]$$
   * **High Influence** ($I_j \approx I_{\max}$) $\to \mu_j \approx 0.08$ (strict, fine-grained adjustments).
   * **Low Influence** ($I_j \approx 0$) $\to \mu_j \approx 0.20$ (broad, exploratory perturbation).

3. **Phase 3: Dynamic Per-Gene Mutation**:
   During offspring generation in the evolutionary loop, individual trajectory scores $e_i$ combine with the sensitivity profile $\mu_j$:
   $$\sigma_{i, j} = \mu_j \cdot (1 - e_i) + 0.02$$
   $$\boldsymbol{\theta}_{\text{child}}[j] = \operatorname{clip}\left(\boldsymbol{\theta}_{\text{parent}}[j] + \mathcal{N}(0, \sigma_{i, j}^2),\, 0,\, 1\right)$$

---

## 4. CCE-TDCCEA (Cooperative Co-Evolutionary TD-CCEA)

### 4.1 Core Problem Addressed
Optimizing a single 20-dimensional monolithic space simultaneously suffers from the **curse of dimensionality** and couples independent economic macro-decisions with immediate tactical micro-decisions.

**CCE-TDCCEA** decomposes the 20D genome into two independent, specialized sub-populations:
1. **Population A (Macro / Economy - 8 Dimensions)**
2. **Population B (Micro / Combat - 12 Dimensions)**

Both sub-populations maintain their own centroids, trajectory histories, archives, and diversity pruning mechanisms, interacting exclusively through **cooperative evaluation ("handshakes")**.

### 4.2 Cooperative Co-Evolution Architecture

```
  ┌─────────────────────────────────┐         ┌─────────────────────────────────┐
  │   Pop A: Macro / Economy (8D)   │         │   Pop B: Micro / Combat (12D)   │
  │   - Population size: N_A = 20   │         │   - Population size: N_B = 20   │
  │   - Sub-centroid: c_A ∈ ℝ⁸      │         │   - Sub-centroid: c_B ∈ ℝ¹²     │
  │   - Sub-archive: A_A (max 15)   │         │   - Sub-archive: A_B (max 15)   │
  └────────────────┬────────────────┘         └────────────────┬────────────────┘
                   │                                           │
                   │               COOPERATIVE HANDSHAKE       │
                   ├─────────────────────────┬─────────────────┤
                   ▼                                           ▼
  ┌─────────────────────────────────┐         ┌─────────────────────────────────┐
  │  Evaluate Macro Agent a_i:      │         │  Evaluate Micro Agent b_j:      │
  │  Combine a_i with Top-3 Micro   │         │  Combine b_j with Top-3 Macro   │
  │  partners from Pop B (top_b)    │         │  partners from Pop A (top_a)    │
  └────────────────┬────────────────┘         └────────────────┬────────────────┘
                   │                                           │
                   ▼                                           ▼
  ┌─────────────────────────────────┐         ┌─────────────────────────────────┐
  │  Compute Macro Trajectory &     │         │  Compute Micro Trajectory &     │
  │  Adaptive Mutation (in ℝ⁸)      │         │  Adaptive Mutation (in ℝ¹²)     │
  └────────────────┬────────────────┘         └────────────────┬────────────────┘
                   │                                           │
                   ▼                                           ▼
  ┌─────────────────────────────────┐         ┌─────────────────────────────────┐
  │  Macro Diversity Pruning (8D)   │         │  Micro Diversity Pruning (12D)  │
  │  & Independent Breed of Pop A   │         │  & Independent Breed of Pop B   │
  └─────────────────────────────────┘         └─────────────────────────────────┘
```

### 4.3 Step-by-Step Execution:

1. **Sub-Agent Decomposition**:
   * `Pop A` instances represent $D_A = 8$ parameters (`MACRO_INDICES = [0,1,2,6,14,15,17,18]`).
   * `Pop B` instances represent $D_B = 12$ parameters (`MICRO_INDICES = [3,4,5,7,8,9,10,11,12,13,16,19]`).

2. **Composite Environment State**:
   $$\mathbf{c}_{\text{env}} = [\mathbf{c}_A \,\|\, \mathbf{c}_B] \in \mathbb{R}^{20}$$

3. **Cooperative Handshake Evaluation**:
   * **Evaluating Macro Agent $a \in \mathcal{P}_A$**:
     Combined with the current **Top-3 Micro agents** (`top_b`):
     $$\boldsymbol{\theta} = \operatorname{combine}(a, b_k) \quad \text{for } b_k \in \operatorname{top}_3(\mathcal{P}_B)$$
     Fitness $f_a$ is the average win rate across these 3 composite matches.
   * **Evaluating Micro Agent $b \in \mathcal{P}_B$**:
     Combined with the current **Top-3 Macro agents** (`top_a`):
     $$\boldsymbol{\theta} = \operatorname{combine}(a_k, b) \quad \text{for } a_k \in \operatorname{top}_3(\mathcal{P}_A)$$
     Fitness $f_b$ is the average win rate across these 3 composite matches.

4. **Independent Sub-Space Trajectory Computation**:
   * Macro distance is computed over $\mathbb{R}^8$ with $d_{\max} = \sqrt{8}$.
   * Micro distance is computed over $\mathbb{R}^{12}$ with $d_{\max} = \sqrt{12}$.
   * Each sub-agent calculates its own trajectory score $e_i$ and adaptive mutation rate $\mu_i = 0.15 \cdot (1 - e_i) + 0.03$.

5. **Sub-Population Diversity Pruning & Dual Archives**:
   * Diversity pruning ($\delta_s = 0.85$) is applied separately on $\mathcal{P}_A$ (8D space) and $\mathcal{P}_B$ (12D space).
   * Separate elite archives $\mathcal{A}_A$ and $\mathcal{A}_B$ retain top macro and micro specialists.
   * Breeding occurs independently within each sub-population.

6. **Composite Solution Assembly**:
   The final evolved strategy is assembled by combining the top macro elite with the top micro elite:
   $$\boldsymbol{\theta}^* = \operatorname{combine}(\operatorname{best}(\mathcal{P}_A),\, \operatorname{best}(\mathcal{P}_B))$$

---

## 5. Comprehensive Comparison Matrix

| Dimension | **TDCCEA** | **SA-TDCCEA** | **CCE-TDCCEA** |
| :--- | :--- | :--- | :--- |
| **Search Space** | Monolithic (1 population $\times$ 20D) | Monolithic (1 population $\times$ 20D) | Decomposed (Pop A: 8D, Pop B: 12D) |
| **Population Size** | $N = 20$ | $N = 20$ | $N_A = 20, \, N_B = 20$ ($40$ sub-agents total) |
| **Trajectory Memory** | Centroid $\mathbf{c}_t \in \mathbb{R}^{20}$, History $H_i$ | Centroid $\mathbf{c}_t \in \mathbb{R}^{20}$, History $H_i$ | Sub-centroids $\mathbf{c}_A \in \mathbb{R}^8, \mathbf{c}_B \in \mathbb{R}^{12}$, independent $H_A, H_B$ |
| **Mutation Strategy** | Dynamic scalar $\mu_i$ (uniform across 20 genes) | Dynamic vector $\boldsymbol{\sigma}_i = \boldsymbol{\mu}_{\text{sensitivity}} \cdot (1 - e_i) + \epsilon$ | Dynamic scalar $\mu_i$ per sub-agent applied across sub-genome |
| **Sensitivity Tuning** | No | Yes (offline parameter influence profiling) | No (relies on structural domain separation) |
| **Evaluation Method** | Direct individual game runs | Direct individual game runs | Cooperative "Handshake" with Top-$K$ cross-partners |
| **Diversity Pruning** | Monolithic 20D distance ($\delta_s = 0.85$) | Monolithic 20D distance ($\delta_s = 0.85$) | Independent Sub-space pruning (8D for A, 12D for B) |
| **Elite Archive** | Single Archive $\mathcal{A}$ ($A_{\max} = 15$) | Single Archive $\mathcal{A}$ ($A_{\max} = 15$) | Dual Archives $\mathcal{A}_A, \mathcal{A}_B$ ($A_{\max} = 15$ each) |
| **Primary Advantage** | Trajectory-guided exploration & memory | Protects sensitive strategic weights from destructive mutation | Resolves curse of dimensionality; specializes Macro and Micro |

---

## 6. Code Mapping Reference

| Algorithm / Component | Source File | Key Classes & Functions |
| :--- | :--- | :--- |
| **TD-CCEA Core** | [`SEM7_PR1_Results.py`](./SEM7_PR1_Results.py) | `Agent`, `EvolveCoreState`, `step_evolve_core` |
| **SA-TDCCEA** | [`SEM7_PR1_Results.py`](./SEM7_PR1_Results.py), [`SA_TDCCEA.py`](../SA_TDCCEA.py) | `run_sensitivity_analysis`, `Agent(pwm=...)`, `step_evolve_core` |
| **CCE-TDCCEA** | [`SEM7_PR1_Results.py`](./SEM7_PR1_Results.py), [`CCE_TDCCEA.py`](../CCE_TDCCEA.py) | `SubAgent`, `CCEAState`, `combine_subagents`, `subpop_survival`, `step_ccea` |
| **Dynamic Map & BFS** | [`SEM7_PR1_Results.py`](./SEM7_PR1_Results.py) | `bfs_reachable`, `pick_wall_position`, `add_walls_for_generation` |
| **Tournament & Eval** | [`SEM7_PR1_Results.py`](./SEM7_PR1_Results.py) | `ParallelGamePool`, `run_head_to_head`, `wilcoxon_test` |
