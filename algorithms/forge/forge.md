# FORGE: Federated Online Reinforcement Graph Evolution  
## Algorithm Design Document  
> IEEE JSTSP Special Issue — Autonomous and Evolutive Optimization in Networked AI  
> Submission Deadline: June 15, 2026

---

## 1. Motivation

### 1.1 Problem Context

Vehicular networks in B5G/6G environments are inherently **non-stationary**: vehicle density, mobility patterns, channel conditions, and interference profiles shift continuously across time and space. Existing deep reinforcement learning (DRL) approaches for resource management — including prior hierarchical and graph-based formulations — suffer from two critical limitations:

1. **Static policy degradation**: Models trained offline or in stationary environments fail silently when deployment conditions drift, with no mechanism to detect or compensate for distributional shift.
2. **Centralized bottleneck**: Centralizing raw observations or gradients across geographically distributed base stations (BSs) and roadside units (RSUs) incurs unacceptable communication overhead and introduces single points of failure.

### 1.2 Gap Analysis

| Limitation | Existing Work | FORGE's Solution |
|---|---|---|
| Non-stationary environments | Static or periodically retrained models | Online drift detection + adaptive policy update |
| Centralized training | FedAvg on flat model weights | Hierarchical federated graph aggregation |
| Ignores topology | MLP or CNN-based DRL | Hierarchical graph attention (intra-cell + inter-cell) |
| Single-level decisions | Flat MARL | Macro (BS-level) + Micro (UE-level) hierarchy |
| No evolutive mechanism | Frozen reward structures | Pseudo-label and reward co-evolution online |

### 1.3 Research Questions

- **RQ1**: Can a hierarchical graph attention structure better capture both local (intra-cell) and global (inter-cell) interference dependencies compared to flat MARL?
- **RQ2**: Can online drift detection trigger targeted federated policy updates without full model retransmission?
- **RQ3**: Does evolutive pseudo-label generation improve sample efficiency in sparse-reward vehicular scenarios?

---

## 2. System Model

### 2.1 Network Topology

Consider a B5G vehicular network with:
- $M$ Base Stations (BSs) / RSUs, each serving as a **macro-agent**
- $K_m$ User Equipments (UEs) / vehicles under BS $m$, each as a **micro-agent**
- A dynamic graph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$ where nodes are BSs/UEs and edges encode interference and communication links
- Time-varying channel coefficients $h_{k,m}^t$ subject to Doppler shifts and path loss

### 2.2 Optimization Objective

Maximize the long-term network utility under Quality of Service (QoS) constraints:

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t \sum_{m=1}^{M} \sum_{k=1}^{K_m} r_{k,m}^t \right]$$

Subject to:
- Latency: $d_{k,m}^t \leq d_{\max}$
- Power: $p_{k,m}^t \leq P_{\max}$
- Spectrum: $\sum_k b_{k,m}^t \leq B_m$

---

## 3. FORGE Algorithm

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FORGE Framework                       │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐ │
│  │  Drift      │───▶│  Evolutive   │──▶│  Federated  │ │
│  │  Detector   │    │  Reward Gen  │   │  Aggregator │ │
│  └─────────────┘    └──────────────┘   └─────────────┘ │
│         │                  │                  │          │
│  ┌──────▼──────────────────▼──────────────────▼──────┐  │
│  │         Hierarchical Graph Attention MARL          │  │
│  │   ┌─────────────────┐   ┌─────────────────────┐   │  │
│  │   │  Macro Layer    │   │   Micro Layer        │   │  │
│  │   │  (BS-level GAT) │   │   (UE-level GAT)     │   │  │
│  │   │  Inter-cell     │   │   Intra-cell         │   │  │
│  │   │  coordination   │   │   scheduling         │   │  │
│  │   └─────────────────┘   └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

#### Component 1: Hierarchical Graph Attention (HiGAT) Encoder

Two-level graph attention:

**Micro-level** (intra-cell): For BS $m$, compute UE embeddings:

$$\mathbf{e}_{k}^{\text{micro}} = \text{GAT}_{\text{micro}}\left(\{s_{k'}^t : k' \in \mathcal{N}(k)\}\right)$$

**Macro-level** (inter-cell): Aggregate BS-level context:

$$\mathbf{e}_{m}^{\text{macro}} = \text{GAT}_{\text{macro}}\left(\{\mathbf{e}_{m'}^{\text{pool}} : m' \in \mathcal{N}(m)\}\right)$$

where $\mathbf{e}_{m}^{\text{pool}} = \text{Readout}(\{\mathbf{e}_k^{\text{micro}} : k \in \mathcal{K}_m\})$.

#### Component 2: Graph-Masked Autoencoder Change-Point Detector (GMAE-CPD)

Rather than classical kernel statistics, FORGE employs a **two-stage neural drift detector** combining self-supervised graph representation learning with Bayesian online change-point inference — designed to capture both structural and feature-level distributional shifts in the dynamic vehicular graph.

##### Stage 1: Graph Masked Autoencoder (GMAE) — Representation Learning

Inspired by GraphMAE (Hou et al., NeurIPS 2022) and adapted for temporal vehicular graphs, a **GMAE encoder-decoder** $(\mathcal{E}_\phi, \mathcal{D}_\phi)$ is trained online via masked reconstruction:

At each timestep $t$, a random mask $\mathcal{M}_t$ is applied to a fraction $\rho \in [0.3, 0.5]$ of node features and edges in $\mathcal{G}_t$:

$$\tilde{\mathcal{G}}_t = \text{Mask}(\mathcal{G}_t, \mathcal{M}_t)$$

The encoder maps the masked graph to a latent representation:

$$\mathbf{Z}_t = \mathcal{E}_\phi(\tilde{\mathcal{G}}_t) \in \mathbb{R}^{N_t \times d}$$

The decoder reconstructs the masked node features using a **scaled cosine error** (more stable than MSE for normalized embeddings):

$$\mathcal{L}_{\text{GMAE}}^t = \frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} \left(1 - \frac{\mathbf{x}_i \cdot \hat{\mathbf{x}}_i}{\|\mathbf{x}_i\| \|\hat{\mathbf{x}}_i\|}\right)$$

The per-node reconstruction error $\ell_i^t = 1 - \cos(\mathbf{x}_i, \hat{\mathbf{x}}_i)$ serves as a **semantically rich anomaly score**: nodes whose channel/mobility context has shifted from learned patterns yield elevated $\ell_i^t$.

The graph-level drift signal is aggregated via a learnable attention-weighted pooling:

$$\xi_t = \sum_{i \in \mathcal{V}_t} a_i^t \cdot \ell_i^t, \quad a_i^t = \text{softmax}(\mathbf{w}_a^\top \mathbf{z}_i^t)$$

This yields a **scalar drift score** $\xi_t$ that reflects both the magnitude and the spatial distribution of anomalous nodes in the network graph.

##### Stage 2: Neural Bayesian Online Change-Point Detection (NBOCD)

Rather than comparing $\xi_t$ against a fixed threshold (which is fragile in non-stationary environments), FORGE maintains a **posterior distribution over run-lengths** $r_t$ — the number of timesteps since the last change point — inspired by Adams & MacKay (2007) but with a learned neural predictive model replacing the parametric hazard function.

**Run-length posterior update:**

$$P(r_t | \xi_{1:t}) \propto \sum_{r_{t-1}} P(\xi_t | r_{t-1}, \xi_{(t-r_{t-1}):t}) \cdot P(r_t | r_{t-1}) \cdot P(r_{t-1} | \xi_{1:t-1})$$

The predictive likelihood $P(\xi_t | r_{t-1}, \cdot)$ is parameterized by a lightweight **Temporal Convolutional Network (TCN)** $f_\psi$ trained online:

$$\hat{\xi}_t = f_\psi(\xi_{t-L:t-1}), \quad P(\xi_t | r_{t-1}) = \mathcal{N}(\hat{\xi}_t,\ \sigma_\psi^2(\xi_{t-L:t-1}))$$

where $\sigma_\psi^2$ is a learned heteroscedastic variance head — allowing the detector to be less sensitive during naturally volatile periods (e.g., rush-hour density spikes) and more sensitive during stable periods.

**Hazard function:** The transition probability $P(r_t = 0 | r_{t-1})$ (i.e., a change point occurred) uses a learned neural hazard:

$$h_t = \sigma(\mathbf{w}_h^\top [\mathbf{z}_t^{\text{pool}}; r_{t-1}])$$

where $\mathbf{z}_t^{\text{pool}}$ is the GMAE pooled embedding, giving the hazard access to structural network context — not just scalar history.

**Drift severity classification:** The posterior mode run-length $\hat{r}_t = \arg\max_{r} P(r_t | \xi_{1:t})$ and the change-point probability $P(r_t = 0 | \xi_{1:t})$ together classify drift severity:

| Condition | Classification | Triggered Action |
|---|---|---|
| $P(r_t=0) < \epsilon_0$ | **No drift** | Normal policy update |
| $\epsilon_0 \leq P(r_t=0) < \epsilon_1$ | **Soft drift** | Local GMAE fine-tune + SAC replay re-weighting |
| $P(r_t=0) \geq \epsilon_1$ and $\hat{r}_t < \tau_{\text{short}}$ | **Hard drift** | Federated aggregation round triggered |
| $P(r_t=0) \geq \epsilon_1$ and $\xi_t > \xi_{\text{struct}}$ | **Structural drift** | Graph topology re-indexing + full federated reset |

The thresholds $\{\epsilon_0, \epsilon_1, \tau_{\text{short}}, \xi_{\text{struct}}\}$ are set once during warm-up and remain fixed, making GMAE-CPD **threshold-light** compared to MMD variants.

##### GMAE Online Update Strategy

To prevent catastrophic forgetting while adapting to drift, the GMAE is updated via **experience replay with drift-weighted sampling**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GMAE}}^{\text{current}} + \lambda_{\text{mem}} \cdot \mathcal{L}_{\text{GMAE}}^{\text{replay}}$$

where replay samples are drawn from a **ring buffer** $\mathcal{B}_{\text{mem}}$ of size $N_{\text{mem}}$, prioritized by their historical $\xi$ scores — ensuring rare but important drift events are retained.

#### Component 3: Evolutive Reward and Pseudo-Label Generator (ERPG)

Rather than a fixed reward function, FORGE evolves reward signals online:

$$r_{k,m}^t = \alpha_t \cdot r_{\text{throughput}} + \beta_t \cdot r_{\text{latency}} + \gamma_t \cdot r_{\text{fairness}}$$

Weights $\{\alpha_t, \beta_t, \gamma_t\}$ are updated via a **meta-reward controller** that maximizes the smoothed performance improvement $\Delta \bar{R}_t$.

Pseudo-labels are generated for unlabeled state transitions using the current policy's value estimate:

$$\tilde{y}_{k,m}^t = V_{\psi}(s_{k,m}^t) + \hat{r}_{k,m}^t$$

#### Component 4: Hierarchical Federated Aggregation (HFA)

FORGE uses a **two-tier federated scheme** aligned with the macro/micro hierarchy:

1. **Intra-cluster aggregation**: BSs within a geographic cluster share micro-policy weights:
$$\theta_m^{\text{micro}} \leftarrow \sum_{m' \in \mathcal{C}_m} w_{m'} \theta_{m'}^{\text{micro}}$$

2. **Global macro aggregation**: Cluster heads share macro-policy weights with a central coordinator (or gossip protocol in fully decentralized mode):
$$\theta^{\text{macro}} \leftarrow \sum_{c=1}^{C} w_c \theta_c^{\text{macro}}$$

Aggregation is triggered **on-demand** by the drift detector, not on a fixed schedule — reducing communication overhead.

### 3.3 Soft Actor-Critic (SAC) Policy Optimization

Each agent optimizes the entropy-regularized objective:

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_t \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot | s_t))\right)\right]$$

Actor, critic, and value networks are updated using standard SAC updates, conditioned on HiGAT embeddings $[\mathbf{e}_k^{\text{micro}} \| \mathbf{e}_m^{\text{macro}}]$.

### 3.4 Full FORGE Algorithm (Pseudocode)

```
Algorithm: FORGE
Input:  G_0 (initial graph), π_0 (initial policy),
        GMAE (E_φ, D_φ), TCN hazard model f_ψ,
        thresholds {ε_0, ε_1, τ_short, ξ_struct}
Output: Evolutive policy π*

Initialize:
  HiGAT encoder, SAC actors/critics, ERPG weights {α,β,γ}
  GMAE warm-up: pretrain (E_φ, D_φ) on G_0 for W_warmup steps
  NBOCD: P(r_0) = 1, ring buffer B_mem = ∅

────────────────────────────────────────────────────
MAIN LOOP
────────────────────────────────────────────────────
For each episode e = 1, ..., E:
  Reset graph G_t = G_0, run-length posterior P(r_t)

  For each timestep t:

    ── PERCEPTION ──
    1. [OBSERVE]   Each agent (k,m) observes local state s_{k,m}^t
                   Update dynamic graph G_t = (V_t, E_t)

    ── REPRESENTATION ──
    2. [ENCODE]    Compute HiGAT embeddings:
                     e_micro ← GAT_micro(intra-cell neighbors of k)
                     e_macro ← GAT_macro(inter-cell BS pool of m)

    ── DRIFT DETECTION (GMAE-CPD) ──
    3. [MASK]      Sample mask M_t, form G̃_t = Mask(G_t, M_t)
    4. [RECONSTRUCT]
                   Z_t ← E_φ(G̃_t)
                   X̂_t ← D_φ(Z_t)
                   Compute per-node cosine error ℓ_i^t
                   Compute drift score ξ_t = Σ a_i^t · ℓ_i^t

    5. [BAYESIAN CPD]
                   Predict ξ̂_t, σ²_t ← f_ψ(ξ_{t-L:t-1})
                   Update run-length posterior P(r_t | ξ_{1:t})
                   using neural hazard h_t = σ(w_h^T [z_pool; r_{t-1}])
                   Compute change-point prob CP_t = P(r_t=0 | ξ_{1:t})

    6. [CLASSIFY DRIFT]
                   If CP_t < ε_0          → no_drift
                   If ε_0 ≤ CP_t < ε_1   → soft_drift
                   If CP_t ≥ ε_1 and r̂_t < τ_short  → hard_drift
                   If CP_t ≥ ε_1 and ξ_t > ξ_struct  → structural_drift

    ── ACTION ──
    7. [ACT]       Sample a_{k,m}^t ~ π(· | e_micro ∥ e_macro)

    ── REWARD EVOLUTION ──
    8. [EVOLVE REWARD]
                   Update ERPG weights {α_t, β_t, γ_t} via meta-controller
                   Compute r_{k,m}^t = α_t·r_tput + β_t·r_lat + γ_t·r_fair

    ── STORAGE ──
    9. [STORE]     Add (s, a, r, s', ξ_t) to local replay buffer B_m
                   Add G_t to ring buffer B_mem (prioritized by ξ_t)

    ── LOCAL UPDATE ──
    10. [UPDATE POLICY]
                   Sample mini-batch from B_m
                   Update SAC actor, critic, value networks
                   Generate pseudo-labels for sparse transitions via ERPG

    11. [UPDATE GMAE]
                   If soft_drift or hard_drift:
                     Fine-tune (E_φ, D_φ) on current + replay batch
                     L_total = L_GMAE^current + λ_mem · L_GMAE^replay
                   Update TCN f_ψ with new (ξ_{t-L:t-1}, ξ_t) pair

    ── FEDERATED STEP ──
    12. [FEDERATED AGGREGATION] (if hard_drift or structural_drift)
                   Intra-cluster: aggregate micro-policy weights θ^micro
                   Global:        aggregate macro-policy weights θ^macro
                   If structural_drift: re-index graph topology, reset NBOCD
                   Broadcast updated weights to all BSs

  End For
End For

Return π* = current evolved policy
────────────────────────────────────────────────────
```

---

## 4. Evaluation Plan

### 4.1 Simulation Environment

| Parameter | Value |
|---|---|
| Simulator | SUMO (mobility) + NS3 or custom gym environment |
| Area | 2km × 2km urban grid |
| Number of BSs (M) | 9–16 |
| Vehicles per BS | 10–50 (dynamic) |
| Bandwidth | 100 MHz (sub-6GHz) + 400 MHz (mmWave) |
| Channel model | 3GPP TR 38.901 Urban Macro |
| Drift injection | Sudden (step), gradual (ramp), periodic (sine) |
| Mobility model | SUMO highway + urban mixed traces |

### 4.2 Baselines

| Baseline | Description |
|---|---|
| **DDPG-Centralized** | Centralized deep deterministic policy gradient |
| **MADDPG** | Multi-agent DDPG, no graph structure |
| **FedAvg-SAC** | Standard FedAvg with flat SAC, no hierarchy |
| **HiGAT-MASAC** | Prior work (no federated, no drift detection) |
| **DQN-Static** | Tabular/DQN with fixed reward, no evolution |
| **FORGE-NoDrift** | Ablation: FORGE without ODD component |
| **FORGE-NoEvo** | Ablation: FORGE without ERPG component |

### 4.3 Experiment Design

**Experiment 1 — Stationary Performance**
- Standard vehicular network with fixed density
- Compare throughput, latency, fairness against all baselines

**Experiment 2 — Non-Stationary Robustness**
- Inject drift events: sudden vehicle surge, RSU failure, channel model shift
- Measure performance degradation and recovery speed

**Experiment 3 — Scalability**
- Vary M ∈ {4, 9, 16, 25} BSs and K ∈ {10, 20, 50} vehicles/BS
- Measure training convergence time and communication overhead

**Experiment 4 — Federated Efficiency**
- Compare on-demand (FORGE) vs. fixed-interval aggregation
- Measure rounds triggered, bytes transmitted, vs. accuracy

**Experiment 5 — Ablation Study**
- FORGE vs. FORGE-NoDrift vs. FORGE-NoEvo vs. FORGE-NoHierarchy
- Isolate contribution of each component

---

## 5. Evaluation Metrics

### 5.1 Network Performance
| Metric | Definition | Target |
|---|---|---|
| **Sum Throughput** | $\sum_{k,m} R_{k,m}^t$ (Mbps) | Maximize |
| **Average Latency** | $\bar{d} = \frac{1}{K}\sum_{k} d_k^t$ (ms) | Minimize |
| **Jain's Fairness Index** | $\mathcal{F} = \frac{(\sum R_k)^2}{K \sum R_k^2}$ | Maximize (→1) |
| **Outage Probability** | $P(R_k < R_{\min})$ | Minimize |

### 5.2 Adaptation & Evolution
| Metric | Definition | Target |
|---|---|---|
| **Drift Detection Latency** | Time steps to detect shift | Minimize |
| **Recovery Time** | Steps to return to pre-drift performance ±5% | Minimize |
| **Policy Regret** | $\sum_t (R_t^* - R_t^{\text{FORGE}})$ | Minimize |
| **Reward Evolution Gain** | $\Delta R$ from ERPG vs. fixed reward | Maximize |

### 5.3 Federated Efficiency
| Metric | Definition | Target |
|---|---|---|
| **Communication Rounds** | Total aggregation rounds triggered | Minimize |
| **Bytes Transmitted** | Total model parameters exchanged (MB) | Minimize |
| **Convergence Episodes** | Episodes to reach 95% of peak performance | Minimize |
| **Heterogeneity Robustness** | Variance in BS-level performance | Minimize |

---

## 6. Key References

### Foundational MARL & Graph RL
1. Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*.
2. Jiang, J., & Lu, Z. (2021). Learning attentional communication for multi-agent cooperation. *NeurIPS*.
3. Veličković, P., et al. (2018). Graph attention networks. *ICLR*.

### Federated Reinforcement Learning
4. Liang, X., et al. (2019). Federated transfer reinforcement learning for autonomous driving. *arXiv:1910.06001*.
5. McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
6. Liu, B., et al. (2022). Federated graph neural networks: Overview, applications, and challenges. *IEEE TNNLS*.

### Vehicular Network Optimization
7. Ye, H., et al. (2019). Deep reinforcement learning based resource allocation for V2V communications. *IEEE TITS*.
8. Cui, Y., et al. (2022). Multi-agent reinforcement learning-based resource allocation for UAV networks. *IEEE JSAC*.
9. He, Y., et al. (2021). Integrated networking, caching, and computing for connected vehicles. *IEEE Communications Magazine*.

### Non-Stationary RL & Neural Drift Detection
10. Padakandla, S., et al. (2021). Reinforcement learning in non-stationary environments. *ACM Computing Surveys*.
11. Hou, Z., et al. (2022). GraphMAE: Self-supervised masked graph autoencoders. *KDD 2022*.
12. Adams, R.P., & MacKay, D.J.C. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
13. Bai, S., et al. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271* (TCN foundation).
14. Ditzler, G., et al. (2015). Learning in nonstationary environments: A survey. *IEEE CIM*.
15. Lu, J., et al. (2023). Concept drift detection via neural predictive coding. *IEEE TNNLS*.

### Hierarchical RL
16. Nachum, O., et al. (2018). Data-efficient hierarchical reinforcement learning. *NeurIPS*.
17. Kulkarni, T.D., et al. (2016). Hierarchical deep reinforcement learning. *NeurIPS*.

### Evolutive / Online Learning in Networks
18. Sun, Y., et al. (2023). Online learning for adaptive signal processing in networked AI. *IEEE JSTSP*.
19. Shi, Y., et al. (2022). Networked AI for 6G: Architecture and challenges. *IEEE Network*.

---

## 7. Novelty Summary

| Contribution | Prior Art | FORGE |
|---|---|---|
| Graph topology | Flat MARL | Hierarchical intra+inter-cell GAT |
| Federated scheme | Fixed-round FedAvg | On-demand drift-triggered aggregation |
| Reward structure | Fixed handcrafted | Evolutive meta-controlled reward |
| Non-stationarity | Ignored or periodic retrain | GMAE-CPD: neural masked autoencoder + Bayesian online change-point posterior |
| Pseudo-supervision | Fully offline | Online pseudo-label generation |

---

*Document version: v1.1 — March 2026*  
*Contact: vinh.dq4@buv.edu.vn*