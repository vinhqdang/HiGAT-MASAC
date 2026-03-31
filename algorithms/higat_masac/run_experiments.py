import os
import subprocess
import argparse

def main():
    algorithms = ['higat_masac', 'mappo', 'maddpg', 'gnn_ddqn', 'random', 'greedy']
    seeds = [42, 43, 44]
    
    os.makedirs('results/higat_masac', exist_ok=True)
    
    for algo in algorithms:
        for seed in seeds:
            print(f"Running {algo} with seed {seed}...")
            # Run train.py with --seed (will need to add --seed to train.py)
            cmd = f"conda run -n py313 python algorithms/higat_masac/train.py --algo {algo} --seed {seed}"
            subprocess.run(cmd, shell=True, check=True)
            
    print("All multi-seed experiments finished!")
    print("Running evaluation...")
    subprocess.run("conda run -n py313 python algorithms/higat_masac/evaluate.py --multi-seed", shell=True, check=True)

if __name__ == "__main__":
    main()
