# HiGAT-MASAC: Implementation Code Plan

**Project:** Hierarchical Graph Attention Multi-Agent Soft Actor-Critic for Joint Resource Optimization in B5G Vehicular Networks  
**Target venue:** Elsevier *Computer Communications* — Special Issue: Next-Generation IoT Urban Computing and Mobile Networks for Sustainable and Smart Mobility  
**Submission deadline:** 30 April 2026  
**This document:** Developer implementation guide — algorithm, data, evaluation, and reporting plan

---

## 1. Motivation and Research Context

### Why this problem matters

Urban vehicular networks in the B5G/6G era must simultaneously satisfy three tightly coupled resource demands:

- **Spectrum efficiency** — V2I and V2V links compete for limited sub-bands; poor allocation degrades throughput for all vehicles in the cell
- **Energy efficiency** — vehicle on-board units are battery-constrained; excessive transmit power drains energy and amplifies interference
- **Computation offloading** — latency-critical tasks (ADAS, HD map updates, infotainment rendering) must be split intelligently between the vehicle CPU and MEC servers at RSUs

These three dimensions are **not independent**. Offloading more to MEC requires more uplink transmission, consuming spectrum and energy. Reducing power saves energy but lowers the offloading data rate, forcing more local computation. No closed-form optimal policy exists under dynamic channel conditions and vehicle mobility.

### Why existing approaches fall short

| Approach | Limitation |
|---|---|
| Classical optimization (convex, Lyapunov) | Requires perfect CSI; too slow for millisecond-scale decisions |
| Single-agent DRL (DDPG, SAC, PPO) | Centralized; does not scale with number of vehicles |
| Standard MARL (MADDPG, MAPPO) | Flat state representation; cannot generalize across topology changes |
| GNN + DRL (GNN-DDQN, GAPO) | Single resource dimension; no hierarchical coordination |
| Federated DRL | Communication overhead; no topology-aware graph encoding |

### Why HiGAT-MASAC is the answer

**Three design choices working together:**

1. **Hierarchical decomposition** — RSU macro-agents decide spectrum blocks and MEC allocation (slow timescale, 100 ms); vehicle micro-agents decide power and offloading ratio (fast timescale, 10 ms). This reduces per-agent action space dramatically and enables scalability.

2. **Graph Attention Networks (GAT)** — interference relationships between vehicles are encoded as a dynamic graph with learnable attention weights. Unlike GraphSAGE or basic GCN, GAT can assign higher weight to high-interference or high-bandwidth neighbors without manual feature engineering. This enables policies that generalize across arbitrary vehicular topologies.

3. **Soft Actor-Critic (SAC)** — the maximum-entropy RL objective explicitly encourages exploration in the hybrid discrete-continuous action space (discrete sub-band selection + continuous power and offloading ratio). SAC outperforms DDPG, TD3, and PPO empirically in high-dimensional continuous action spaces with sparse rewards, as confirmed in recent IoV literature.

---

## 2. Key Related Work (for Developer Context)

The developer should understand how HiGAT-MASAC relates to the following foundational papers. These will also become the baseline implementations and bibliography.

### 2.1 MARL for V2X resource management

**Liang et al. (2019)** — "Spectrum Sharing in Vehicular Networks Based on Multi-Agent Reinforcement Learning," *IEEE JSAC*  
→ Seminal fingerprint-based multi-agent DQN for V2V spectrum sharing. **Baseline B1.**

**Wang et al. (2025)** — "Multi-Agent DRL for V2X Resource Allocation: Disentangling Challenges and Benchmarking Solutions," *arXiv:2603.06607*  
→ Systematic benchmark of 8 MARL algorithms; MAPPO is best; topology generalization identified as the key bottleneck. **Baseline B2 (MAPPO).**

**Tan et al. (2025)** — "Enhanced Multi-Agent Deep Reinforcement Learning for Efficient Task Offloading and Resource Allocation in Vehicular Networks," *Vehicular Communications*  
→ Attention-enhanced MADDPG for joint offloading + resource allocation. **Baseline B3 (MADDPG).**

### 2.2 GNN for wireless resource allocation

**Ji et al. (2025)** — "Graph Neural Networks and Deep Reinforcement Learning Based Resource Allocation for V2X Communications," *IEEE IoT Journal*  
→ GraphSAGE + DDQN for distributed V2X resource allocation. Treats all neighbors equally. **Baseline B4 (GNN-DDQN).**

**GAPO (2025)** — "A Graph Attention-Based RL Algorithm for Congestion-Aware Task Offloading in Multi-Hop VEC," *Sensors*  
→ GAT + PPO for offloading only. No spectrum or energy. **Closest structural relative; not a full baseline.**

### 2.3 SAC for IoV

**Qi et al. (2024)** — "DRL-Based AoI-Aware Resource Allocation for RIS-Aided IoV Networks," *arXiv:2406.11245*  
→ SAC outperforms DDPG/TD3/PPO for AoI-aware vehicular resource allocation. Validates SAC choice.

### 2.4 Hierarchical DRL for networks

**Peng & Shen (2020)** — "Deep Reinforcement Learning Based Resource Management for Multi-Access Edge Computing in Vehicular Networks," *IEEE Trans. Network Science and Engineering*  
→ Hierarchical DDPG for joint spectrum/computing/storage. Static MEC. **Baseline B5 (H-DDPG).**

---

## 3. Algorithm Description

### 3.1 System Model

#### Network setup
- $M$ RSUs, each co-located with a MEC server
- $N$ vehicles moving in a Manhattan grid urban scenario
- $K$ orthogonal sub-bands of bandwidth $B$ MHz each
- V2I links: vehicle uploads task to RSU MEC via uplink
- V2V links: direct vehicle-to-vehicle sidelink (underlay, reuses V2I spectrum)

#### Channel model
- Path loss: 3GPP UMi model (TR 38.901), LoS/NLoS conditions
- Shadowing: log-normal, σ = 4 dB
- Fast fading: Rayleigh (NLoS V2V), Rician K=3 dB (LoS V2I)
- SINR calculation accounts for inter-vehicle interference on shared sub-bands

#### Task model
- Each vehicle generates tasks: `(data_size_bits, cpu_cycles_per_bit, max_delay_ms)`
- Partial offloading: vehicle offloads fraction `α ∈ [0,1]` to MEC, executes `(1-α)` locally
- Local CPU: fixed per vehicle; MEC CPU: allocated dynamically per task

#### Latency model
```
T_local   = (1-α) * data_size * cpu_cycles / local_cpu_freq
T_uplink  = α * data_size / uplink_rate
T_mec     = α * data_size * cpu_cycles / mec_cpu_alloc
T_total   = max(T_local, T_uplink + T_mec)   # parallel local + edge execution
```

#### Energy model
```
E_comm = transmit_power * T_uplink
E_comp = kappa * local_cpu_freq^2 * (1-α) * data_size * cpu_cycles
E_total = E_comm + E_comp   (kappa = 1e-28)
```

### 3.2 Optimization Problem

**Decision variables:**
- `X[n,k] ∈ {0,1}` — sub-band k assigned to vehicle n (discrete)
- `p[n,k] ∈ [0, P_max]` — transmit power of vehicle n on sub-band k (continuous)
- `α[n] ∈ [0,1]` — offloading ratio of vehicle n (continuous)
- `f[n,m] ≥ 0` — MEC CPU allocated to vehicle n at RSU m (continuous)

**Objective (minimize weighted sum):**
```
minimize  w1 * sum(T_n) + w2 * sum(E_n) - w3 * sum(throughput_n)
```

**Constraints:**
- C1: Each vehicle uses at most K_max sub-bands
- C2: No two vehicles in the same cluster share a sub-band (intra-cluster orthogonality)
- C3: Transmit power within [0, P_max]
- C4: Offloading ratio within [0, 1]
- C5: Total MEC CPU allocation per RSU within capacity
- C6: Task delay within deadline T_max for each vehicle

This is a **Mixed-Integer Nonlinear Program (MINLP)** — NP-hard, intractable at millisecond scale. Reformulated as a **Dec-POMDP** and solved with MARL.

### 3.3 HiGAT-MASAC Architecture

#### Two-tier hierarchy

```
MACRO LEVEL  (RSU agents, 100 ms timescale)
  Input:  cluster-level interference graph
  Decide: sub-band assignment X[n,k], MEC CPU allocation f[n,m]
  Output: resource envelopes broadcast to vehicles

MICRO LEVEL  (vehicle agents, 10 ms timescale)
  Input:  local neighborhood graph + resource envelope from macro
  Decide: transmit power p[n,k], offloading ratio α[n]
  Output: executed actions per slot
```

Coordination is **one-directional**: macro → micro via resource envelope. No feedback loop during execution (reduces control overhead).

#### Graph construction

**Macro graph** (per RSU, per macro slot):
```
Nodes:    vehicles in cluster C_m
Node features: [avg_channel_gain, queue_length, data_size, cpu_intensity,
                max_delay, predicted_speed, predicted_heading]
Edges:    vehicle pairs within distance threshold d_int = 150 m
Edge weights: interference channel gain h[n,n']
```

**Micro graph** (per vehicle, per micro slot):
```
Nodes:    K-nearest neighbors of vehicle n
Node features: [local_CSI, residual_deadline, queue_state, battery_level]
Edges:    distance-based adjacency
Edge weights: interference levels
```

#### GAT module (shared across tiers via knowledge distillation)

```python
# Multi-head GAT (H heads, L layers)
# For each head h and neighbor pair (n, n'):
e[n,n'] = LeakyReLU( a[h].T @ concat(W[h] @ x_n, W[h] @ x_n') )
alpha[n,n'] = softmax over neighbors of e[n,n']
z_n = ELU( concat over H heads of: sum over neighbors of alpha[n,n'] * W[h] @ x_n' )

# Output: node embedding z_n ∈ R^(H * d')
# Graph-level macro state: mean pooling of {z_n} for RSU cluster
```

Hyperparameters to tune: H (number of heads, try 4/8), d' (embedding dim, try 64/128), L (GAT layers, try 2/3).

#### SAC at macro level (RSU agents)

```
State s_m:       mean-pooled GAT embedding of cluster
Action a_m:      (X[n,k] via Gumbel-Softmax, f[n,m] continuous)
Reward r_m:      -w1*sum(T_n) + w3*sum(throughput_n) - w_vio*sum(delay_violations)
Networks:        Actor (policy), Twin Q-networks (critics), auto-tuned temperature β
```

#### SAC at micro level (vehicle agents)

```
State s_n:       local GAT embedding concatenated with resource envelope e_n
Action a_n:      (p[n,k] continuous, α[n] continuous)
Reward r_n:      -w1*T_n - w2*E_n + w3*throughput_n - w_vio*delay_violation
Networks:        Actor, Twin Q-networks, auto-tuned temperature β
```

#### Key SAC equations (implement these exactly)

```
# Critic update (twin Q):
L(Q) = E[ (Q(s,a) - (r + γ * (min_Q_target(s',a') - β*log π(a'|s'))))^2 ]

# Actor update:
L(π) = E[ β*log π(a|s) - min_Q(s,a) ]

# Temperature update:
L(β) = E[ -β * (log π(a|s) + H_target) ]
H_target = -dim(action_space)   # target entropy
```

#### Training procedure

```
CTDE: Centralized Training, Decentralized Execution

for episode in 1..E_max:
    reset environment, load SUMO mobility trace
    for macro_slot t in 1..T_mac:
        for each RSU m:
            build cluster graph G_m
            compute GAT embeddings {z_n_mac}
            sample macro action a_m ~ π_m_mac
            broadcast resource envelopes to vehicles
        
        for micro_slot t in 1..T_mic:
            for each vehicle n:
                build local graph G_n
                compute GAT embeddings z_n_mic
                sample micro action (p, α) ~ π_n_mic
                execute, observe reward and next state
                store transition in micro replay buffer
            
            update micro policies from micro buffer (every T_update steps)
        
        collect macro rewards, store macro transitions
        update macro policies from macro buffer
    
    every T_KD episodes:
        knowledge distillation: align GAT parameters across macro and micro tiers
```

### 3.4 Implementation Stack

```
Language:        Python 3.10+
RL framework:    PyTorch (custom SAC implementation preferred over stable-baselines3
                 for full control of actor/critic architecture)
GNN library:     PyTorch Geometric (torch_geometric) — use GATConv layer
Simulation:      SUMO (vehicular mobility) + custom Python gym environment
                 OR CityFlow if SUMO integration is complex
Channel model:   Custom Python (3GPP UMi formulas) OR use SimPy for event-driven sim
Logging:         Weights & Biases (wandb) for experiment tracking
Plotting:        Matplotlib + Seaborn
```

**Suggested project structure:**
```
higat_masac/
├── env/
│   ├── vehicular_env.py       # OpenAI Gym-compatible Dec-POMDP environment
│   ├── channel_model.py       # 3GPP UMi path loss, fading, SINR
│   ├── task_model.py          # Task generation, latency, energy computation
│   └── sumo_interface.py      # SUMO mobility trace loader
├── models/
│   ├── gat_encoder.py         # Multi-head GAT (torch_geometric GATConv)
│   ├── sac_macro.py           # SAC for RSU macro-agents (hybrid action space)
│   ├── sac_micro.py           # SAC for vehicle micro-agents (continuous)
│   └── higat_masac.py         # Full hierarchical framework, coordinates tiers
├── baselines/
│   ├── mappo.py               # Baseline B2
│   ├── maddpg.py              # Baseline B3
│   ├── gnn_ddqn.py            # Baseline B4
│   ├── h_ddpg.py              # Baseline B5
│   └── random_greedy.py       # Baseline B0 (random + greedy)
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── configs/
│   └── default.yaml           # All hyperparameters
└── results/
    └── plots/
```

---

## 4. Dataset and Evaluation Plan

### 4.1 Simulation Environment Setup

There is no standard public dataset for this problem. The evaluation environment is **fully simulated** using established standards, which is the norm in this literature. The simulation must be reproducible and parameterized.

#### Mobility model
- **Tool:** SUMO (Simulation of Urban MObility) — https://eclipse.dev/sumo/
- **Scenario:** Manhattan grid, 9 city blocks, each block 250 m × 250 m, 3-lane roads
- **Traffic density:** low (20 vehicles), medium (50 vehicles), high (80–100 vehicles)
- **Vehicle speed:** 30–60 km/h urban, sampled from truncated normal distribution
- **Simulation duration:** 600 seconds per episode, 10 ms step resolution
- **Seed:** fix 5 random seeds and average results for statistical reliability

If SUMO integration is too complex for the first version, use a **simplified random waypoint mobility model** in Python (acceptable for conference-level work; SUMO preferred for journal).

#### Network parameters

| Parameter | Value |
|---|---|
| Number of RSUs (M) | 4 |
| Sub-bands (K) | 10 |
| Sub-band bandwidth | 1 MHz |
| Max vehicle transmit power P_max | 23 dBm |
| Noise power σ² | -114 dBm |
| MEC CPU capacity per RSU F_mec | 20 GHz |
| Local vehicle CPU f_loc | 1 GHz |
| Path loss model | 3GPP UMi (TR 38.901) |
| Shadowing std dev | 4 dB |
| Max delay per task T_max | 100 ms |
| Task data size d | Uniform [0.5, 2] Mbit |
| CPU intensity c | Uniform [500, 1500] cycles/bit |
| Objective weights (w1, w2, w3) | (0.4, 0.3, 0.3) |

#### Training hyperparameters (starting point, tune with grid search)

| Parameter | Value |
|---|---|
| Replay buffer size | 100,000 |
| Batch size | 256 |
| Discount factor γ | 0.99 |
| Actor/Critic learning rate | 3e-4 |
| Soft target update τ | 0.005 |
| GAT heads H | 4 |
| GAT embedding dim d' | 64 |
| GAT layers L | 2 |
| Macro slot duration | 100 ms |
| Micro slot duration | 10 ms |
| Training episodes | 2000 |
| Evaluation interval | every 50 episodes |

### 4.2 Baselines to Implement

| ID | Method | Description |
|---|---|---|
| B0-Random | Random allocation | Random sub-band assignment, fixed power, random offloading ratio |
| B0-Greedy | Greedy | Max-SINR sub-band, max power, full offloading |
| B1 | MA-DQN | Multi-agent DQN with fingerprints (Liang et al. 2019) |
| B2 | MAPPO | Multi-agent PPO with centralized critic (current SOTA benchmark) |
| B3 | MADDPG | Multi-agent DDPG (Tan et al. 2025 variant) |
| B4 | GNN-DDQN | GraphSAGE + DDQN (Ji et al. 2025) |
| B5 | H-DDPG | Hierarchical DDPG without GAT (Peng & Shen 2020 adapted) |
| Ablation A1 | HiGAT-MASAC w/o GAT | Replace GAT with MLP encoder (flat state) |
| Ablation A2 | HiGAT-MASAC w/o hierarchy | Single-tier flat MASAC with GAT |
| Ablation A3 | HiGAT-MASAC w/o SAC | Replace SAC with PPO at both tiers |
| **Proposed** | **HiGAT-MASAC** | Full hierarchical GAT + SAC framework |

### 4.3 Evaluation Metrics

Compute all metrics averaged over the last 100 evaluation episodes across 5 random seeds. Report mean ± standard deviation.

#### Primary metrics (must report)

| Metric | Formula / Description | Unit |
|---|---|---|
| Average task completion delay | Mean T_n across all vehicles and episodes | ms |
| Total system energy | Sum of E_n across all vehicles per episode | Joules |
| V2I sum-rate capacity | Sum of r_{n,m} across all V2I links | Mbps |
| V2V payload delivery ratio | Fraction of V2V tasks delivered within T_max | % |
| Delay QoS violation rate | Fraction of tasks with T_n > T_max | % |

#### Secondary metrics (report in additional analysis)

| Metric | Description |
|---|---|
| Convergence speed | Number of training episodes to reach 95% of final reward |
| Scalability curve | All primary metrics vs. vehicle count N ∈ {20, 40, 60, 80, 100} |
| Energy efficiency | V2I sum-rate / total energy (bits per Joule) |
| Jain's fairness index | Fairness of delay distribution across vehicles |
| Control overhead | Number of macro→micro messages per second |
| Inference time | Average policy decision latency per agent (ms) — must be < 10 ms for practicality |

### 4.4 Experimental Scenarios

Run the following scenarios to support different paper claims:

| Scenario | Purpose | What varies |
|---|---|---|
| S1: Standard urban | Main comparison table | All baselines, N=50 vehicles |
| S2: Scalability | Claim: HiGAT-MASAC scales | N ∈ {20, 40, 60, 80, 100} |
| S3: Weight sensitivity | Claim: flexible trade-off | (w1,w2,w3) sweeps |
| S4: Mobility impact | Claim: robust to speed | Vehicle speed ∈ {20, 40, 60, 80} km/h |
| S5: Ablation study | Claim: each component contributes | A1, A2, A3 vs. full model |
| S6: GAT head analysis | Claim: attention is meaningful | H ∈ {1, 2, 4, 8} |

---

## 5. Report (Paper Structure and Content Plan)

The paper targets approximately **12–14 pages** in Elsevier double-column format. The following sections must be included.

### 5.1 Abstract (~250 words)
Must state:
- Problem: joint spectrum/power/offloading in B5G vehicular networks
- Gap: no hierarchical GAT-MASAC for this problem exists
- Method: HiGAT-MASAC, two-tier hierarchy, GAT topology encoding, SAC optimization
- Key results: 3 headline numbers (delay reduction %, energy reduction %, throughput gain %) vs. best baseline
- Venue alignment: smart mobility, B5G, AI-based autonomous management

### 5.2 Introduction (~1.5 pages)
Structure:
1. Opening paragraph: urban mobility challenge, B5G requirements
2. Three-resource coupling problem explanation with concrete example
3. Why classical optimization fails
4. Why existing DRL/MARL/GNN approaches are insufficient (brief gap table)
5. Proposed solution and key design choices (3 bullet points)
6. List of contributions (4 numbered items)
7. Paper organization paragraph

### 5.3 Related Work (~1.5 pages)
Four subsections:
1. DRL for vehicular resource management
2. Graph neural networks for wireless networks
3. Multi-agent RL for V2X
4. Hierarchical RL for network management

Each subsection: 2–3 paragraphs. End each with a sentence explicitly stating what is missing and how this paper addresses it.

### 5.4 System Model (~1.5 pages)
Five subsections (as defined in Section 3.1 above):
1. Network topology (with a system diagram figure — **Figure 1**)
2. Channel model with SINR formula
3. Computation task and MEC model
4. Latency model
5. Energy model

### 5.5 Problem Formulation (~0.5 pages)
- State the MINLP problem formally with all constraints
- Argue NP-hardness briefly (cite relevant complexity result)
- Motivate Dec-POMDP reformulation

### 5.6 HiGAT-MASAC Framework (~3 pages)
Five subsections:
1. Hierarchical architecture overview (with architecture diagram — **Figure 2**)
2. Graph construction (macro and micro, with a graph visualization example — **Figure 3**)
3. GAT encoding equations
4. SAC at macro level (state/action/reward/network specification)
5. SAC at micro level (state/action/reward/network specification)
6. Training algorithm (pseudocode — **Algorithm 1**)

### 5.7 Simulation Setup (~0.5 pages)
- Table of all simulation parameters (**Table 1**)
- Brief description of SUMO mobility scenario
- Baseline implementation notes

### 5.8 Evaluation Results (~3 pages)

**Figure and table plan:**

| Item | Content |
|---|---|
| Table 2 | Main comparison: all baselines vs. HiGAT-MASAC on all primary metrics (S1) |
| Figure 4 | Convergence curves: reward vs. training episode for all methods |
| Figure 5 | Scalability: 5 primary metrics vs. vehicle count N (line plots, S2) |
| Figure 6 | Ablation study: bar chart comparing A1, A2, A3, full model (S5) |
| Figure 7 | Trade-off surface: delay vs. energy vs. throughput under weight sweep (S3) |
| Figure 8 | Attention weight visualization: heatmap of GAT attention on example graph (qualitative insight) |
| Table 3 | Inference time and control overhead per method (practical feasibility) |

**Narrative structure for results section:**
1. Overall performance (Table 2): lead with the headline numbers, explain why HiGAT-MASAC wins on each metric
2. Convergence analysis (Figure 4): SAC's entropy regularization enables faster and more stable convergence than DDPG/PPO variants
3. Scalability (Figure 5): GAT's local message-passing architecture maintains performance as N grows; flat MARL degrades
4. Ablation (Figure 6): each component contributes; removing GAT hurts the most under high density
5. Trade-off analysis (Figure 7): the three objectives are controllable via weight vector
6. Attention visualization (Figure 8): qualitative evidence that attention correctly emphasizes high-interference neighbors

### 5.9 Discussion (~0.5 pages)
Address:
- Practical deployment considerations (inference latency, communication overhead)
- Limitations: assumes synchronous macro/micro slots; does not model RSU failures
- Broader applicability beyond vehicular networks (e.g., UAV swarms, IoT edge networks)

### 5.10 Conclusion (~0.3 pages)
- One-sentence problem statement
- One-sentence solution summary
- Three headline results
- One sentence on future work (e.g., extend to RIS-assisted B5G, add semantic communication layer)

### 5.11 References
Target 35–45 references. Must include:
- All baselines listed in Section 2 above
- 3GPP TR 38.901 (channel model standard)
- Original GAT paper: Veličković et al. (2018), ICLR
- Original SAC paper: Haarnoja et al. (2018), ICML
- Original MARL survey or benchmark (Wang et al. 2025)
- At least 3–4 references from *Computer Communications* or *Vehicular Communications* (Elsevier) to signal venue fit

---

## Appendix: Development Milestones

| Milestone | Deliverable | Estimated effort |
|---|---|---|
| M1 | Vehicular gym environment (channel model + task model + SUMO interface) | 1 week |
| M2 | GAT encoder (torch_geometric, macro + micro graphs) | 3 days |
| M3 | SAC micro-agent (continuous action space) | 3 days |
| M4 | SAC macro-agent (hybrid discrete-continuous, Gumbel-Softmax) | 4 days |
| M5 | Full HiGAT-MASAC integration + CTDE training loop | 4 days |
| M6 | All 5 baselines + ablation variants | 1 week |
| M7 | Experiment runner: all 6 scenarios, 5 seeds, logging | 3 days |
| M8 | Figures, tables, result analysis | 3 days |
| **Total** | | **~5 weeks** |

**Critical path:** M1 (environment) must be complete before any RL training can begin. Validate the environment first by running a random policy and confirming reward signals are numerically stable and constraints are correctly enforced.

---

*Document prepared for HiGAT-MASAC implementation — Computer Communications special issue submission, deadline 30 April 2026.*