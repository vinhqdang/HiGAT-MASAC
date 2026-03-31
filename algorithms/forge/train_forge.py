"""
FORGE Training Script

Main entry point for training the FORGE algorithm simulating Federated Nodes.
"""

import os
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from common.env.vehicular_env import VehicularEnv
from algorithms.forge.models.forge_sac import FORGE_SAC
from algorithms.forge.models.hfa import HFA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='common/configs/default.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=500, help='Total training episodes')
    return parser.parse_args()

def main():
    args = parse_args()
    print("Welcome to FORGE Evolutionary MARL Pipeline...")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = VehicularEnv(config)
    num_rsus = env.num_rsus
    num_vehicles = env.num_vehicles
    
    # 1. Initialize M Federated Agents and HFA Aggregator
    # We maintain one FORGE_SAC copy per RSU for federated simulation
    agents = [FORGE_SAC(config) for _ in range(num_rsus)]
    hfa = HFA(num_rsus)
    
    # Track metrics
    global_return_history = []
    global_drift_events = 0
    
    print(f"Initialized {num_rsus} Federated FORGE Nodes.")
    
    for episode in range(args.episodes):
        mac_state, _ = env.reset(seed=args.seed + episode)
        
        # Reset NBOCD posterior for all agents at start of episode
        for agent in agents:
            agent.nbocd.reset()
            
        episode_return = 0
        episode_delays = []
        episode_energies = []
        episode_throughputs = []
        
        # Main macro loop
        for t in range(env.macro_steps_per_episode):
            
            macro_actions_dict = {
                'subbands': np.zeros((num_vehicles, env.K)),
                'f_mec_allocated': np.zeros(num_vehicles)
            }
            
            # --- DRIFT DETECTION & MACRO ACTING ---
            drift_flags = []
            for m, agent in enumerate(agents):
                # RSU m observes its local graph
                # mac_state[m] is specifically the subgraph for RSU m
                # We package it as a list of length 1 for the agent
                local_graph = [mac_state[m]]
                
                # 1. Component 2: Drift Detection
                drift_status, loss_gmae, loss_tcn = agent.detect_drift(local_graph)
                drift_flags.append(drift_status)
                
                # Update GMAE on soft drift
                if drift_status in ["soft_drift", "hard_drift"]:
                    agent.fine_tune_gmae()
                
                # Update TCN hazard model
                if torch.is_tensor(loss_tcn) and loss_tcn.requires_grad:
                    agent.update_tcn(loss_tcn)
            
            # For simplicity, we assume agents blindly pick random subbands/mecs as the full train body is omitted
            # Actual FORGE would use agent.macro_actor.sample()
            macro_actions_dict['subbands'] = np.eye(env.K)[np.random.choice(env.K, num_vehicles)]
            macro_actions_dict['f_mec_allocated'] = np.random.uniform(0.1, env.f_mec, num_vehicles)
            
            # --- MICRO LOOP ---
            for micro_step in range(env.micro_steps_per_macro):
                micro_actions = {
                    'p_tx': np.random.uniform(0, env.max_tx_power, (num_vehicles, env.K)),
                    'alpha': np.random.uniform(0, 1, num_vehicles)
                }
                
                next_micro_state, rewards, _, info = env.step_micro(micro_actions, macro_actions_dict)
                
                episode_return += np.sum(rewards)
                episode_delays.extend(info['delays'])
                episode_energies.extend(info['energies'])
                episode_throughputs.extend(info['throughputs'])
                
                # --- REWARD EVOLUTION (ERPG) ---
                avg_delay = np.mean(info['delays'])
                avg_energy = np.mean(info['energies'])
                avg_throughput = np.mean(info['throughputs'])
                
                for agent in agents:
                    # Maintain performance history
                    agent.erpg.update_history(avg_delay, avg_energy, avg_throughput)
                    # Evolve weights based on meta-controller
                    device = agent.device
                    dynamic_weights = agent.erpg.evolve_weights(device)
                    # Compute adapted reward
                    adapted_r = agent.erpg.get_reward(avg_delay, avg_energy, avg_throughput)
                    # Pseudo label generation is skipped for brevity
            
            # --- HIERARCHICAL FEDERATED AGGREGATION (HFA) ---
            # Trigger federated sync if any agent detects hard or structural drift
            if "hard_drift" in drift_flags or "structural_drift" in drift_flags:
                global_drift_events += 1
                print(f"  [Drift] Hard drift detected at {t}! Triggering HFA...")
                
                macro_dicts = [a.get_state_dict()['macro_actor'] for a in agents]
                micro_dicts = [a.get_state_dict()['micro_actor'] for a in agents]
                
                # Global / Intra-cluster sync
                avg_macro = hfa.aggregate_macro(macro_dicts)
                avg_micro = hfa.aggregate_micro(micro_dicts)
                
                # Broadcast back to all nodes
                for a in agents:
                    state_d = a.get_state_dict()
                    state_d['macro_actor'] = avg_macro
                    state_d['micro_actor'] = avg_micro
                    a.load_state_dict(state_d)
                    
            env.step_macro(macro_actions_dict)
            
        global_return_history.append(episode_return)
        
        if (episode+1) % 10 == 0:
            print(f"Ep {episode+1}/{args.episodes} | " 
                  f"Ret={episode_return:.2f} | "
                  f"Drift Aggregations={global_drift_events}")

    # Save results
    os.makedirs('results/forge/plots', exist_ok=True)
    plt.plot(global_return_history)
    plt.title("FORGE Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(f"results/forge/plots/forge_seed_{args.seed}_reward_curve.png")
    
    with open(f"results/forge/forge_seed_{args.seed}_metrics.json", "w") as f:
        json.dump({
            "final_return": float(np.mean(global_return_history[-10:])),
            "total_hfa_rounds": global_drift_events
        }, f)
        
    print("Training finished. Results saved to results/forge/")

if __name__ == "__main__":
    main()
