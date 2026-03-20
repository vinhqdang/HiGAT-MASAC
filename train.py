import yaml
import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from env.vehicular_env import VehicularEnv
from models.higat_masac import HiGAT_MASAC
from baselines.random_greedy import RandomGreedyBaseline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--algo', type=str, default='higat_masac', choices=['higat_masac', 'random', 'greedy'])
    parser.add_argument('--test-env', action='store_true', help='Test environment step logic without training')
    parser.add_argument('--dummy-run', action='store_true', help='Test 2 episodes only for pipeline verification')
    return parser.parse_args()

def train(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    env = VehicularEnv(config)
    
    if args.algo == 'higat_masac':
        agent = HiGAT_MASAC(config)
    else:
        agent = RandomGreedyBaseline(config)

    episodes = 2 if args.dummy_run else config['rl']['training_episodes']
    t_kd = config['rl']['t_kd_episodes']
    eval_interval = config['rl']['eval_interval']
    
    macro_steps = 2 if args.dummy_run else env.macro_steps_per_episode
    micro_steps = env.micro_steps_per_macro
    
    reward_history = []
    
    for episode in range(episodes):
        macro_state, _ = env.reset(seed=episode)
        ep_reward = 0.0
        
        for t_mac in range(macro_steps):
            # 1. Macro tier execution
            macro_actions = {'subbands': [], 'f_mec_allocated': []}
            
            # Since macro_state is a list of graphs (one per RSU), we iterate or batch
            # For simplicity, we assume central or batched decision.
            # We'll mock the graph input logic here. In practice, map features to edge_index and call network.
            
            # Mock generating states as dummy tensors for the agent just to verify pipeline shape compilation
            # In a real run, construct Torch Geometric Data objects from macro_state list.
            for m in range(env.num_rsus):
                pass
                
            # Assume we got actions from macro agent
            if args.algo == 'higat_masac':
                # Dummy tensors for compilation validation
                dummy_macro = (torch.zeros((env.num_vehicles, 5)), torch.zeros((2, 0), dtype=torch.long), None, torch.zeros(env.num_vehicles, dtype=torch.long)) 
                subband, fmec = agent.select_macro_action(dummy_macro)
            elif args.algo == 'random':
                subband, fmec = agent.select_macro_action_random(macro_state)
            elif args.algo == 'greedy':
                subband, fmec = agent.select_macro_action_greedy(macro_state)
                
            macro_actions['subbands'] = subband
            macro_actions['f_mec_allocated'] = fmec
            
            # 2. Micro tier execution
            micro_state_list = env.get_micro_state(macro_actions)
            
            for t_mic in range(micro_steps):
                p_tx = np.zeros((env.num_vehicles, env.K))
                alpha = np.zeros(env.num_vehicles)
                
                # Each vehicle decides its micro action
                for n in range(env.num_vehicles):
                    if args.algo == 'higat_masac':
                        dummy_micro = (torch.zeros((1, 4)), torch.zeros((2, 0), dtype=torch.long), None, torch.zeros(1, dtype=torch.long))
                        action = agent.select_micro_action(dummy_micro)
                        # action is [p_tx_scalar, alpha_scalar]
                        p_val, alpha_val = action
                        # map back from [-1, 1] to [0, max]
                        alpha[n] = (alpha_val + 1) / 2.0
                        p_tx[n, np.argmax(subband[n])] = (p_val + 1) / 2.0 * env.max_tx_power
                    else:
                        act = agent.select_micro_action_random(micro_state_list) if args.algo == 'random' else agent.select_micro_action_greedy(micro_state_list)
                        alpha[n] = act[1]
                        p_tx[n, np.argmax(subband[n])] = act[0] * env.max_tx_power
                
                micro_actions = {'p_tx': p_tx, 'alpha': alpha}
                next_micro_state_list, rewards, done, info = env.step_micro(micro_actions, macro_actions)
                
                ep_reward += np.sum(rewards)
                
                if args.algo == 'higat_masac':
                    # Add to micro replay buffer
                    # Using dummy shapes for pipeline
                    agent.micro_buffer.add(
                        np.zeros(config['rl']['gat_embed_dim']),
                        np.zeros(2),
                        np.sum(rewards)/env.num_vehicles,
                        np.zeros(config['rl']['gat_embed_dim']),
                        0
                    )
                    agent.train_micro()
                    
                micro_state_list = next_micro_state_list
                
            # After macro slot, add transition to macro buffer
            if args.algo == 'higat_masac':
                agent.macro_buffer.add(
                    np.zeros(config['rl']['gat_embed_dim']),
                    subband,
                    fmec,
                    np.sum(rewards)/env.num_vehicles,
                    np.zeros(config['rl']['gat_embed_dim']),
                    0
                )
                agent.train_macro()
        
        if args.algo == 'higat_masac' and episode % t_kd == 0:
            agent.knowledge_distillation()
            
        print(f"Episode {episode}: Reward = {ep_reward:.2f}")
        reward_history.append(ep_reward)
        
        # Save checkpoints or plot periodically...
        if episode > 0 and episode % eval_interval == 0 and args.algo == 'higat_masac':
            os.makedirs('results/plots', exist_ok=True)
            plt.plot(reward_history)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig("results/plots/reward_curve.png")
            plt.close()
            
    print("Training Complete!")
    
if __name__ == "__main__":
    args = parse_args()
    if args.test_env:
        # Just run environment loops to verify shapes doesn't crash
        args.algo = 'random'
        args.dummy_run = True
    train(args)
