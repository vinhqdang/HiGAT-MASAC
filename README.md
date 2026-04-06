# HiGAT-MASAC & FORGE

This repository contains two research algorithms for resource optimization in B5G/6G vehicular networks, both under active development and review.

---

## 📦 Algorithms

### 1. HiGAT-MASAC
**Hierarchical Graph Attention Multi-Agent Soft Actor-Critic**

Joint resource optimization (spectrum, power, MEC) for stationary vehicular networks using hierarchical GAT and continuous SAC with knowledge distillation.

> 📄 Manuscript: `manuscripts/higat_masac/`

### 2. FORGE *(Federated Online Reinforcement Graph Evolution)*
**IEEE JSTSP submission — Deadline: June 15, 2026**

Extends HiGAT-MASAC with three additional components for **non-stationary** B5G/6G environments:
- **GMAE-CPD**: Graph Masked Autoencoder + Bayesian Online Change-Point Detector — detects drift severity (`no_drift` / `soft` / `hard` / `structural`)
- **ERPG**: Evolutive Reward & Pseudo-label Generator — dynamically adapts reward weights α/β/γ online
- **HFA**: Hierarchical Federated Aggregation — drift-triggered (not fixed-schedule) federated sync of macro/micro policies

> 📄 Design doc: `algorithms/forge/forge.md`

---

## 🗂️ Repository Structure

```
HiGAT-MASAC/
├── algorithms/
│   ├── higat_masac/         # HiGAT-MASAC algorithm
│   │   ├── models/          # GATEncoder, MicroSAC, MacroSAC, HiGAT-MASAC trainer
│   │   └── train.py         # Training script
│   └── forge/               # FORGE algorithm
│       ├── models/
│       │   ├── gmae_cpd.py  # Graph MAE + NBOCD drift detector
│       │   ├── erpg.py      # Evolutive reward meta-controller
│       │   ├── hfa.py       # Hierarchical federated aggregation
│       │   └── forge_sac.py # Integrated FORGE-SAC agent
│       ├── train_forge.py           # FORGE training pipeline
│       ├── evaluate_forge.py        # Evaluation script
│       ├── compare_nonstationarity.py  # FORGE vs HiGAT-MASAC benchmark
│       └── forge.md                 # Algorithm design document
├── common/
│   ├── env/                 # Shared vehicular environment
│   │   ├── vehicular_env.py # Two-timescale Dec-POMDP orchestrator
│   │   ├── channel_model.py # 3GPP UMi path loss + fading
│   │   ├── task_model.py    # MEC workload + delay/energy model
│   │   └── mobility_model.py# Manhattan grid mobility
│   ├── baselines/           # MADDPG, MAPPO, GNN-DDQN, Random, Greedy
│   └── configs/
│       └── default.yaml     # Shared configuration
├── manuscripts/
│   ├── higat_masac/         # HiGAT-MASAC paper (under review)
│   └── forge/               # FORGE paper (in preparation)
├── results/
│   ├── higat_masac/         # Training curves, metrics, plots
│   └── forge/               # FORGE results + comparison plots
└── requirements.txt
```

---

## ⚙️ Environment Setup

```bash
conda create -n py313 python=3.13 -y
conda activate py313
pip install -r requirements.txt
```

**Key dependencies**: PyTorch, PyTorch Geometric, numpy, matplotlib, PyYAML

---

## 🚀 Running the Pipelines

### HiGAT-MASAC

```bash
# Full training (200 episodes, stationary environment)
python -m algorithms.higat_masac.train --algo higat_masac --seed 42

# Random / Greedy baselines
python -m algorithms.higat_masac.train --algo random --seed 42
python -m algorithms.higat_masac.train --algo greedy --seed 42

# Quick smoke-test (2 episodes)
python -m algorithms.higat_masac.train --dummy-run --algo higat_masac
```

### FORGE

```bash
# Full FORGE training (200 episodes, with GMAE warm-up)
python -m algorithms.forge.train_forge --episodes 200 --warmup 50 --seed 42

# Evaluate on stationary environment
python -m algorithms.forge.evaluate_forge --eval-episodes 20

# Evaluate under injected drift
python -m algorithms.forge.evaluate_forge --eval-episodes 20 --drift-inject

# Non-stationary benchmark: FORGE vs HiGAT-MASAC
python -m algorithms.forge.compare_nonstationarity
```

---

## 📊 Key Results

### Stationary Performance (200 episodes, seed=42)

| Algorithm | Avg Throughput | Avg Return |
|---|---|---|
| HiGAT-MASAC | 74.6 Mbps | 2,312 |
| FORGE (50 ep) | 71.2 Mbps | 6,593 |

> FORGE's higher return reflects ERPG's dynamic reward adaptation. Throughput is similar at early training — FORGE's advantage is most significant under **non-stationary** conditions.

### Non-Stationary Robustness (Experiment 2)

FORGE vs HiGAT-MASAC under sudden vehicle surge drift at episode 30:

| Metric | HiGAT-MASAC (frozen post-drift) | FORGE (adaptive) |
|---|---|---|
| Pre-drift Avg Tp | — | — |
| Post-drift Avg Tp | degrades, no recovery | adapts via HFA |
| Violation Rate | worsens | mitigated by drift detection |

> See `results/forge/nonstationarity/` for full plots.

---

## 📐 Configuration

All parameters are in `common/configs/default.yaml`:

```yaml
env:
  num_rsus: 4
  num_vehicles: 20
  num_subbands: 10
  subband_bandwidth_mhz: 20.0
  rsu_mec_capacity_ghz: 20.0

rl:
  training_episodes: 200
  batch_size: 256
  gat_embed_dim: 64
  gamma: 0.99
```

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{higat-masac,
  title={Hierarchical Graph Attention Multi-Agent SAC for B5G Vehicular Networks},
  author={Vinh, D.Q. et al.},
  note={Under review}
}

@article{forge2026,
  title={FORGE: Federated Online Reinforcement Graph Evolution for Non-Stationary Vehicular Networks},
  author={Vinh, D.Q. et al.},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2026},
  note={In preparation}
}
```

---

*Contact: vinh.dq4@buv.edu.vn*
