"""
Multi-Scenario Experiment Runner for HiGAT-MASAC
Implements all evaluation scenarios from plan.md:
  S1: Standard urban benchmark (all algorithms, N=50)
  S2: Scalability (N in {20, 40, 60, 80, 100})
  S3: Weight sensitivity (w1, w2, w3 sweeps)
  S4: Mobility impact (vehicle speed)
  S5: Ablation study (No-GAT, No-Hierarchy, No-SAC)
  S6: GAT head analysis (H in {1, 2, 4, 8})
"""
import os
import subprocess
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEEDS = [42, 43, 44]
ALGOS_FULL = ['higat_masac', 'mappo', 'maddpg', 'gnn_ddqn', 'random', 'greedy']
ALGOS_MAIN = ['higat_masac', 'mappo', 'maddpg', 'gnn_ddqn']

def run(algo, seed, overrides=None, out_prefix=''):
    """Run a single train.py experiment if result doesn't exist."""
    tag = f"{out_prefix}_{algo}_seed_{seed}" if out_prefix else f"{algo}_seed_{seed}"
    path = f"results/{tag}_metrics.json"
    if os.path.exists(path):
        print(f"  [SKIP] {path} already exists")
        return

    cmd = f"conda run -n py313 python train.py --algo {algo} --seed {seed}"
    if overrides:
        cmd += " --override " + " ".join(overrides)
    if out_prefix:
        cmd += f" --out-prefix {out_prefix}"
    print(f"  >> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  [WARNING] Experiment failed: {cmd}")

def load_metrics(tag):
    """Load metrics JSON for a given tag (prefix_algo_seedN or algo_seedN)."""
    d_vals, e_vals, t_vals = [], [], []
    for seed in SEEDS:
        path = f"results/{tag}_seed_{seed}_metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            d_vals.append(data['delay'])
            e_vals.append(data['energy'])
            t_vals.append(data['throughput'])
    if not d_vals:
        return None
    return {
        'delay_mean': np.mean(d_vals), 'delay_std': np.std(d_vals),
        'energy_mean': np.mean(e_vals), 'energy_std': np.std(e_vals),
        'throughput_mean': np.mean(t_vals), 'throughput_std': np.std(t_vals),
    }

def save_bar_chart(title, filename, labels, delay_m, delay_s, energy_m, energy_s, tp_m, tp_s):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i, (means, stds, ylabel, color) in enumerate([
        (delay_m,  delay_s,  'Delay (ms)',    'coral'),
        (energy_m, energy_s, 'Energy (J)',    'mediumseagreen'),
        (tp_m,     tp_s,     'Throughput (Mbps)', 'steelblue'),
    ]):
        ax[i].bar(x, means, yerr=stds, capsize=5, color=color)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels, rotation=45, ha='right')
        ax[i].set_ylabel(ylabel)
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/{filename}')
    plt.close()
    print(f"  Saved: results/plots/{filename}")

def save_line_chart(title, filename, x_vals, x_label, series_dict):
    """series_dict: {label: {'delay': [...], 'energy': [...], 'throughput': [...]}}"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    metrics = ['delay_mean', 'energy_mean', 'throughput_mean']
    ylabels = ['Delay (ms)', 'Energy (J)', 'Throughput (Mbps)']
    for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for label, data in series_dict.items():
            y = [d[metric] if d else 0 for d in data]
            ax[i].plot(x_vals, y, marker='o', label=label)
        ax[i].set_xlabel(x_label)
        ax[i].set_ylabel(ylabel)
        ax[i].legend(fontsize=7)
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/{filename}')
    plt.close()
    print(f"  Saved: results/plots/{filename}")


# ─────────────────────────────────────────────
# S1: Standard Urban Benchmark
# ─────────────────────────────────────────────
def run_s1():
    print("\n=== S1: Standard Urban Benchmark (N=50, all algos) ===")
    for algo in ALGOS_FULL:
        for seed in SEEDS:
            run(algo, seed, out_prefix='s1')
    # Plot
    labels, dm, ds, em, es, tm, ts = [], [], [], [], [], [], []
    for algo in ALGOS_FULL:
        m = load_metrics(f's1_{algo}')
        if m:
            labels.append(algo); dm.append(m['delay_mean']); ds.append(m['delay_std'])
            em.append(m['energy_mean']); es.append(m['energy_std'])
            tm.append(m['throughput_mean']); ts.append(m['throughput_std'])
    save_bar_chart('S1: Standard Urban Benchmark (N=50)', 's1_benchmark.png', labels, dm, ds, em, es, tm, ts)


# ─────────────────────────────────────────────
# S2: Scalability Study
# ─────────────────────────────────────────────
def run_s2():
    print("\n=== S2: Scalability (N in {20, 40, 60, 80, 100}) ===")
    vehicle_counts = [20, 40, 60, 80, 100]
    series = {algo: [] for algo in ALGOS_MAIN}
    for N in vehicle_counts:
        for algo in ALGOS_MAIN:
            for seed in SEEDS:
                run(algo, seed, overrides=[f'env.num_vehicles={N}'], out_prefix=f's2_n{N}')
        for algo in ALGOS_MAIN:
            series[algo].append(load_metrics(f's2_n{N}_{algo}'))
    save_line_chart('S2: Scalability vs. Vehicle Count', 's2_scalability.png', vehicle_counts, 'Number of Vehicles', series)


# ─────────────────────────────────────────────
# S3: Weight Sensitivity
# ─────────────────────────────────────────────
def run_s3():
    print("\n=== S3: Weight Sensitivity ===")
    weight_configs = [
        ('balanced',    0.4, 0.3, 0.3),
        ('delay_focus', 0.6, 0.2, 0.2),
        ('energy_focus',0.2, 0.6, 0.2),
        ('thru_focus',  0.2, 0.2, 0.6),
    ]
    labels, dm, ds, em, es, tm, ts = [], [], [], [], [], [], []
    for name, w1, w2, w3 in weight_configs:
        for seed in SEEDS:
            run('higat_masac', seed,
                overrides=[f'env.w1={w1}', f'env.w2={w2}', f'env.w3={w3}'],
                out_prefix=f's3_{name}')
        m = load_metrics(f's3_{name}_higat_masac')
        if m:
            labels.append(name); dm.append(m['delay_mean']); ds.append(m['delay_std'])
            em.append(m['energy_mean']); es.append(m['energy_std'])
            tm.append(m['throughput_mean']); ts.append(m['throughput_std'])
    save_bar_chart('S3: HiGAT-MASAC Weight Sensitivity', 's3_weight_sensitivity.png', labels, dm, ds, em, es, tm, ts)


# ─────────────────────────────────────────────
# S4: Mobility Impact
# ─────────────────────────────────────────────
def run_s4():
    print("\n=== S4: Mobility Impact (speed {20,40,60,80} km/h) ===")
    speeds = [20, 40, 60, 80]
    series = {algo: [] for algo in ALGOS_MAIN}
    for speed in speeds:
        for algo in ALGOS_MAIN:
            for seed in SEEDS:
                run(algo, seed,
                    overrides=[f'env.vehicle_speed_kmh_min={speed}', f'env.vehicle_speed_kmh_max={speed}'],
                    out_prefix=f's4_v{speed}')
        for algo in ALGOS_MAIN:
            series[algo].append(load_metrics(f's4_v{speed}_{algo}'))
    save_line_chart('S4: Performance vs. Vehicle Speed', 's4_mobility.png', speeds, 'Vehicle Speed (km/h)', series)


# ─────────────────────────────────────────────
# S5: Ablation Study
# ─────────────────────────────────────────────
def run_s5():
    print("\n=== S5: Ablation Study ===")
    # We run higat_masac with component-disabling flags:
    # A1: No GAT (set gat_layers=0 → fallback to linear)
    # A2: No hierarchy (set macro_slot_duration_ms == micro_slot_duration_ms to flatten)
    # A3: No SAC (use mappo as the non-SAC multi-agent alternative)
    ablations = [
        ('full',         [], 'higat_masac'),
        ('no_gat',       ['rl.gat_layers=0', 'rl.gat_heads=1'], 'higat_masac'),
        ('no_hierarchy', ['rl.macro_slot_duration_ms=10'], 'higat_masac'),
        ('no_sac',       [], 'mappo'),  # MAPPO as non-SAC MARL baseline
    ]
    labels, dm, ds, em, es, tm, ts = [], [], [], [], [], [], []
    for name, overrides, algo in ablations:
        for seed in SEEDS:
            run(algo, seed, overrides=overrides, out_prefix=f's5_{name}')
        m = load_metrics(f's5_{name}_{algo}')
        if m:
            labels.append(name); dm.append(m['delay_mean']); ds.append(m['delay_std'])
            em.append(m['energy_mean']); es.append(m['energy_std'])
            tm.append(m['throughput_mean']); ts.append(m['throughput_std'])
    save_bar_chart('S5: Ablation Study', 's5_ablation.png', labels, dm, ds, em, es, tm, ts)


# ─────────────────────────────────────────────
# S6: GAT Head Analysis
# ─────────────────────────────────────────────
def run_s6():
    print("\n=== S6: GAT Head Analysis (H in {1,2,4,8}) ===")
    heads = [1, 2, 4, 8]
    series = {'higat_masac': []}
    for H in heads:
        for seed in SEEDS:
            run('higat_masac', seed, overrides=[f'rl.gat_heads={H}'], out_prefix=f's6_h{H}')
        series['higat_masac'].append(load_metrics(f's6_h{H}_higat_masac'))
    save_line_chart('S6: Performance vs. GAT Heads', 's6_gat_heads.png', heads, 'GAT Attention Heads (H)', series)


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Run all HiGAT-MASAC evaluation scenarios')
    parser.add_argument('--scenarios', nargs='+', default=['s1', 's2', 's3', 's4', 's5', 's6'],
                        choices=['s1', 's2', 's3', 's4', 's5', 's6'],
                        help='Which scenarios to run (default: all)')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    scenario_map = {
        's1': run_s1, 's2': run_s2, 's3': run_s3,
        's4': run_s4, 's5': run_s5, 's6': run_s6,
    }
    for s in args.scenarios:
        scenario_map[s]()

    print("\n=== All scenarios complete! Figures saved to results/plots/ ===")


if __name__ == '__main__':
    main()
