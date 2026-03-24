# HiGAT-MASAC

Hierarchical Graph Attention Multi-Agent Soft Actor-Critic for Joint Resource Optimization in B5G Vehicular Networks.

## Research Context
Urban vehicular networks in the B5G/6G era must simultaneously satisfy three tightly coupled resource demands: spectrum efficiency, energy efficiency, and computation offloading. This project implements a hierarchical MARL architecture using GAT to encode interference topology and continuous/hybrid SAC algorithms to allocate power, spectrum subbands, and computational MEC resources.

## Project Structure
- `env/`: OpenGym-compatible Vehicular Simulator
  - `channel_model.py`: 3GPP UMi Path Loss, Shadowing, and Fast Fading
  - `task_model.py`: MEC workload generator and Delay/Energy trackers
  - `sumo_interface.py`: 9-block Manhattan Grid mobility generator
  - `vehicular_env.py`: The core two-timescale Dec-POMDP orchestrator
- `models/`: PyTorch modules (GAT Encoder, Micro SAC, Macro SAC, HiGAT MASAC Integrator)
- `baselines/`: Benchmark models (Random, Greedy, MAPPO, MADDPG, GNN-DDQN)
- `configs/`: YAML Configuration parameters for the RL pipeline
- `results/`: Training artifacts, metrics, and plots

## Environment Setup
It is recommended to run this project in a Conda environment containing PyTorch, Gymnasium, and PyTorch Geometric.

```bash
conda create -n py313 python=3.13 -y
conda activate py313
pip install -r requirements.txt
```

## Running the Pipeline

### Testing the Environment
To verify that the environment runs without any tensor shape anomalies or logical errors:
```bash
python train.py --test-env
```

### Fast Dummy Run
To run a fast pipeline verification (2 episodes, small lengths) to ensure Neural Networks compute gradients properly:
```bash
python train.py --dummy-run
```

### Full Training
To run the full algorithm (configured by `configs/default.yaml`):
```bash
# HiGAT-MASAC Proposed Algorithm
python train.py --algo higat_masac

# Random Baseline
python train.py --algo random

# Greedy Baseline
python train.py --algo greedy
```

### Evaluation
After results are saved in `results/`, generate the comparative graphs:
```bash
python evaluate.py
```
