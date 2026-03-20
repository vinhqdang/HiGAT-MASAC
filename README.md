# HiGAT-MASAC

Hierarchical Graph Attention Multi-Agent Soft Actor-Critic for Joint Resource Optimization in B5G Vehicular Networks.

## Setup
```bash
conda create -n py313 python=3.13
conda activate py313
pip install -r requirements.txt
```

## Running
Test environment:
```bash
python train.py --test-env
```

To run training dummy loop (2 episodes):
```bash
python train.py --dummy-run
```

To evaluate and plot results:
```bash
python evaluate.py
```
