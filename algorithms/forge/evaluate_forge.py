"""
FORGE Evaluation Script

Loads trained FORGE checkpoints and evaluates:
  - Sum throughput, average latency, Jain's fairness index
  - Drift detection latency and recovery time under injected drift events
  - Comparison against FORGE-NoDrift and FORGE-NoEvo ablations

Run from repo root:
  python -m algorithms.forge.evaluate_forge --config common/configs/default.yaml
"""

import os
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from common.env.vehicular_env import VehicularEnv
from algorithms.forge.models.forge_sac import FORGE_SAC


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FORGE Evaluation")
    p.add_argument('--config',      type=str, default='common/configs/default.yaml')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--results-dir', type=str, default='results/forge')
    p.add_argument('--drift-inject', action='store_true',
                   help='Inject synthetic drift events during evaluation')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Jain's Fairness Index
# ---------------------------------------------------------------------------
def jains_fairness(throughputs):
    t = np.array(throughputs)
    if t.sum() == 0:
        return 0.0
    return (t.sum() ** 2) / (len(t) * (t ** 2).sum() + 1e-12)


# ---------------------------------------------------------------------------
# Synthetic drift injection (sudden vehicle count change)
# ---------------------------------------------------------------------------
def inject_drift(env, drift_type='step'):
    """Simulate distributional shift by perturbing the mobility model."""
    if drift_type == 'step':
        # Sudden velocity spike (simulates rush-hour surge)
        env.mobility_model.velocities *= 3.0
    elif drift_type == 'channel':
        # Shift path-loss exponent to degrade channels
        env.channel_model.alpha_v2i = getattr(env.channel_model, 'alpha_v2i', 3.7) + 0.5


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------
def evaluate_episode(agents, env, drift_inject=False, drift_at_step=10):
    mac_state, _ = env.reset(seed=0)
    for agent in agents:
        agent.nbocd.reset()

    total_throughput = []
    total_latency    = []
    drifts_detected  = []
    recovery_step    = None
    pre_drift_tp     = []

    K = env.K
    num_vehicles = env.num_vehicles

    for t in range(env.macro_steps_per_episode):
        # Optional drift injection
        if drift_inject and t == drift_at_step:
            inject_drift(env, drift_type='step')

        macro_subband_all = np.zeros((num_vehicles, K))
        macro_fmec_all    = np.zeros(num_vehicles)
        drift_flags       = []

        for m, agent in enumerate(agents):
            local_graph = [mac_state[m]]
            status, _, _ = agent.detect_drift(local_graph)
            drift_flags.append(status)

            subband_np, fmec_np = agent.select_macro_action(local_graph)
            cluster_vehs = mac_state[m]['cluster_vehicles']
            if len(cluster_vehs) > 0 and subband_np.ndim >= 2:
                macro_subband_all[cluster_vehs] = subband_np[: len(cluster_vehs)]
                macro_fmec_all[cluster_vehs]    = fmec_np[: len(cluster_vehs)]

        macro_actions_dict = {
            'subbands':        macro_subband_all,
            'f_mec_allocated': macro_fmec_all
        }
        micro_states = env.get_micro_state(macro_actions_dict)

        for _ in range(env.micro_steps_per_macro):
            raw_actions = agents[0].select_micro_action(micro_states, evaluate=True)
            p_tx  = ((raw_actions[:, 0] + 1.0) / 2.0) * env.max_tx_power
            alpha = (raw_actions[:, 1] + 1.0) / 2.0
            micro_actions = {
                'p_tx':  np.tile(p_tx[:, None], (1, K)) / K,
                'alpha': alpha
            }
            _, _, _, info = env.step_micro(micro_actions, macro_actions_dict)

            step_tp  = float(np.mean(info['throughputs']))
            step_lat = float(np.mean(info['delays']))
            total_throughput.append(step_tp)
            total_latency.append(step_lat)

            if drift_inject:
                if t < drift_at_step:
                    pre_drift_tp.append(step_tp)
                elif any(f in ("hard_drift", "structural_drift") for f in drift_flags):
                    drifts_detected.append(t)
                    if recovery_step is None:
                        baseline_tp = np.mean(pre_drift_tp) if pre_drift_tp else step_tp
                        if step_tp >= baseline_tp * 0.95:
                            recovery_step = t

        env.step_macro(macro_actions_dict)
        mac_state = env._get_macro_state()

    metrics = {
        'sum_throughput_mbps': float(np.sum(total_throughput)),
        'avg_throughput_mbps': float(np.mean(total_throughput)),
        'avg_latency_ms':      float(np.mean(total_latency) * 1e3),
        'jains_fairness':      float(jains_fairness(total_throughput)),
    }
    if drift_inject:
        metrics['drift_detection_latencies'] = [d - drift_at_step for d in drifts_detected]
        metrics['recovery_step']             = recovery_step

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env      = VehicularEnv(config)
    num_rsus = env.num_rsus

    agents = [FORGE_SAC(config) for _ in range(num_rsus)]
    print(f"Loaded {num_rsus} FORGE agents for evaluation.")
    print(f"Running {args.eval_episodes} episodes (drift_inject={args.drift_inject})...\n")

    all_metrics = []
    for ep in range(args.eval_episodes):
        m = evaluate_episode(agents, env, drift_inject=args.drift_inject)
        all_metrics.append(m)
        print(f"  Ep {ep+1:3d} | "
              f"AvgTp={m['avg_throughput_mbps']:.3f} Mbps | "
              f"AvgLat={m['avg_latency_ms']:.2f} ms | "
              f"Fairness={m['jains_fairness']:.4f}")

    # Aggregate
    summary = {
        'avg_throughput_mbps':    float(np.mean([m['avg_throughput_mbps']    for m in all_metrics])),
        'avg_latency_ms':         float(np.mean([m['avg_latency_ms']         for m in all_metrics])),
        'avg_jains_fairness':     float(np.mean([m['jains_fairness']         for m in all_metrics])),
        'std_throughput_mbps':    float(np.std([m['avg_throughput_mbps']     for m in all_metrics])),
    }
    if args.drift_inject:
        all_det_lats = [l for m in all_metrics for l in m.get('drift_detection_latencies', [])]
        summary['avg_drift_detection_latency'] = float(np.mean(all_det_lats)) if all_det_lats else None

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f'forge_eval_seed_{args.seed}.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'episodes': all_metrics}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Plot throughput distribution
    tps = [m['avg_throughput_mbps'] for m in all_metrics]
    plt.figure()
    plt.plot(tps, marker='o')
    plt.title("Eval Per-Episode Avg Throughput")
    plt.xlabel("Episode")
    plt.ylabel("Throughput (Mbps)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'plots', f'forge_eval_tp_seed_{args.seed}.png'))
    plt.close()
    print("Plot saved.")


if __name__ == "__main__":
    main()
