import subprocess
import os
import argparse
import json
import numpy as np

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def run_multiseed_training(seeds, episodes=200):
    print(f"\n--- MULTI-SEED TRAINING ({len(seeds)} seeds) ---")
    for seed in seeds:
        run_cmd(f"python -m algorithms.forge.train_forge --episodes {episodes} --seed {seed}")

def run_ablations(seeds, episodes=200, warmup=200):
    print(f"\n--- ABLATION STUDIES ---")
    variants = [
        {"name": "full",    "args": ""},
        {"name": "no_cpd",  "args": "--no-cpd"},
        {"name": "no_erpg", "args": "--no-erpg"},
        {"name": "no_hfa",  "args": "--no-hfa"},
    ]
    for v in variants:
        print(f"Variant: {v['name']}")
        save_dir = f"results/forge/ablation/{v['name']}"
        os.makedirs(save_dir, exist_ok=True)
        for seed in seeds:
            run_cmd(f"python -m algorithms.forge.train_forge --episodes {episodes} --seed {seed} --warmup {warmup} --save-dir {save_dir} {v['args']} --no-plot")

def run_scalability(seeds, episodes=100, warmup=200):
    print(f"\n--- SCALABILITY TESTS ---")
    veh_counts = [20, 40, 60, 80, 100]
    for n in veh_counts:
        print(f"Vehicles: {n}")
        for seed in seeds:
            save_dir = f"results/forge/scalability/n_{n}/seed_{seed}"
            os.makedirs(save_dir, exist_ok=True)
            run_cmd(f"python -m algorithms.forge.train_forge --episodes {episodes} --seed {seed} --num-vehicles {n} --warmup {warmup} --save-dir {save_dir} --no-plot")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--test-run', action='store_true', help='Quick 2-ep run for testing')
    args = p.parse_args()

    # Use 5 seeds for high-quality results
    seeds = [42, 123, 789, 101, 202][:args.seeds]
    episodes = 2 if args.test_run else args.episodes
    warmup   = 2 if args.test_run else 100 # Reduced warmup for manuscript efficiency
    
    print(f"Configuration: {len(seeds)} seeds, {episodes} episodes per run")
    
    # 1. Ablation Studies (Full, NoCPD, NoERPG, NoHFA)
    run_ablations(seeds, episodes, warmup)
    
    # 2. Scalability Tests (N=20 to 100)
    run_scalability(seeds, episodes, warmup)

if __name__ == "__main__":
    main()
