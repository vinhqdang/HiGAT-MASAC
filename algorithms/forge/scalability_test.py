import os, yaml, torch, json
import numpy as np
import matplotlib.pyplot as plt
from common.env.vehicular_env import VehicularEnv
from algorithms.forge.models.forge_sac import FORGE_SAC

def run_scalability_test(n_vehicles, episodes=5):
    with open('common/configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override
    config['env']['num_vehicles'] = n_vehicles
    print(f"\n--- Testing Scalability: N={n_vehicles} vehicles ---")
    
    env = VehicularEnv(config)
    agents = [FORGE_SAC(config) for _ in range(env.num_rsus)]
    
    results = {'throughput': [], 'delay': [], 'energy': [], 'gmae_loss': []}
    
    for ep in range(episodes):
        mac_state, _ = env.reset(seed=ep)
        ep_tp, ep_lat, ep_nrg, ep_gl = [], [], [], []
        
        for t in range(env.macro_steps_per_episode):
            # Just use random actions or loaded policy to measure throughput/delay
            macro_subband = np.eye(env.K)[np.random.choice(env.K, n_vehicles)]
            macro_fmec = np.random.uniform(0.1, 1.0, n_vehicles) * env.f_mec
            ma = {'subbands': macro_subband, 'f_mec_allocated': macro_fmec}
            
            # Measure GMAE forward pass time/loss
            for m in range(env.num_rsus):
                lg = [mac_state[m]]
                x, ei, _ = agents[m].pack_macro_graph(lg)
                with torch.no_grad():
                    l, _, _ = agents[m].gmae_cpd(x, ei)
                    ep_gl.append(l.item())

            micro_states = env.get_micro_state(ma)
            for _ in range(env.micro_steps_per_macro):
                # Simple power/offloading action
                p_tx = np.random.uniform(0.1, 1.0, (n_vehicles, env.K)) * env.max_tx_power
                alpha = np.random.uniform(0.1, 0.9, n_vehicles)
                _, _, _, info = env.step_micro({'p_tx': p_tx, 'alpha': alpha}, ma)
                ep_tp.append(np.mean(info['throughputs']))
                ep_lat.append(np.mean(info['delays']))
                ep_nrg.append(np.mean(info['energies']))
            
            env.step_macro(ma)
            mac_state = env._get_macro_state()
            
        results['throughput'].append(np.mean(ep_tp))
        results['delay'].append(np.mean(ep_lat) * 1000)
        results['energy'].append(np.mean(ep_nrg))
        results['gmae_loss'].append(np.mean(ep_gl))
        print(f"  Ep {ep+1}/{episodes} | AvgTp={results['throughput'][-1]:.2f} Mbps")

    summary = {k: float(np.mean(v)) for k, v in results.items()}
    summary['n_vehicles'] = n_vehicles
    return summary

def main():
    os.makedirs('results/forge/scalability', exist_ok=True)
    counts = [20, 40, 60, 80, 100]
    all_summaries = []
    
    for n in counts:
        summary = run_scalability_test(n)
        all_summaries.append(summary)
        
    with open('results/forge/scalability/summary.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(counts, [s['throughput'] for s in all_summaries], marker='o')
    axes[0].set_title('Throughput vs Scalability (Random Policy)')
    axes[0].set_xlabel('Num Vehicles'); axes[0].set_ylabel('Avg Throughput (Mbps)')

    axes[1].plot(counts, [s['gmae_loss'] for s in all_summaries], marker='s', color='orange')
    axes[1].set_title('GMAE Recon Loss vs Graph Size')
    axes[1].set_xlabel('Num Vehicles'); axes[1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('results/forge/scalability/scalability_plot.png')
    print("\nScalability testing finished. Summary saved to results/forge/scalability/")

if __name__ == "__main__":
    main()
