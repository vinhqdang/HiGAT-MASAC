import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-seed', action='store_true', help='Evaluate multi-seed results')
    return parser.parse_args()

def plot_metrics(args):
    os.makedirs('results/plots', exist_ok=True)
    
    algorithms = ['higat_masac', 'mappo', 'maddpg', 'gnn_ddqn', 'random', 'greedy']
    
    delay_means = []
    delay_stds = []
    energy_means = []
    energy_stds = []
    throughput_means = []
    throughput_stds = []
    
    seeds = [42, 43, 44] if args.multi_seed else [42]
    
    with open('results/evaluation_report.txt', 'w') as report:
        report.write("HiGAT-MASAC Evaluation Results (Multi-Seed)\n")
        report.write("===========================================\n")
        
        for algo in algorithms:
            d_vals, e_vals, t_vals = [], [], []
            for seed in seeds:
                filepath = f'results/{algo}_seed_{seed}_metrics.json'
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    d_vals.append(data['delay'])
                    e_vals.append(data['energy'])
                    t_vals.append(data['throughput'])
            
            if len(d_vals) > 0:
                d_m, d_s = np.mean(d_vals), np.std(d_vals)
                e_m, e_s = np.mean(e_vals), np.std(e_vals)
                t_m, t_s = np.mean(t_vals), np.std(t_vals)
            else:
                d_m, d_s = 0.0, 0.0
                e_m, e_s = 0.0, 0.0
                t_m, t_s = 0.0, 0.0
                
            delay_means.append(d_m)
            delay_stds.append(d_s)
            energy_means.append(e_m)
            energy_stds.append(e_s)
            throughput_means.append(t_m)
            throughput_stds.append(t_s)
            
            report.write(f"Algorithm: {algo}\n")
            report.write(f"  Avg Delay: {d_m:.2f} ± {d_s:.2f} ms\n")
            report.write(f"  Total Energy: {e_m:.2f} ± {e_s:.2f} J\n")
            report.write(f"  Sum Throughput: {t_m:.2f} ± {t_s:.2f} Mbps\n\n")

    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].bar(x, delay_means, width, yerr=delay_stds, capsize=5, color='orange')
    ax[0].set_title('Average Completion Delay')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(algorithms, rotation=45)
    ax[0].set_ylabel('ms')
    
    ax[1].bar(x, energy_means, width, yerr=energy_stds, capsize=5, color='green')
    ax[1].set_title('Total System Energy')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(algorithms, rotation=45)
    ax[1].set_ylabel('Joules')
    
    ax[2].bar(x, throughput_means, width, yerr=throughput_stds, capsize=5, color='blue')
    ax[2].set_title('V2I Sum-Rate Capacity')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(algorithms, rotation=45)
    ax[2].set_ylabel('Mbps')
    
    plt.tight_layout()
    plt.savefig('results/plots/primary_metrics.png')
    plt.close()
    

if __name__ == "__main__":
    args = parse_args()
    
    print("Running evaluation plotting...")
    plot_metrics(args)
    print("Check results/plots/ and results/evaluation_report.txt")
