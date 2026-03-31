import yaml
import argparse
import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from common.env.vehicular_env import VehicularEnv
from algorithms.higat_masac.models.higat_masac import HiGAT_MASAC
from common.baselines.random_greedy import RandomGreedyBaseline
from common.baselines.mappo import MAPPOAgent
from common.baselines.maddpg import MADDPGAgent
from common.baselines.gnn_ddqn import GNNDDQNAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='common/configs/default.yaml', help='Path to config file')
    parser.add_argument('--algo', type=str, default='higat_masac', choices=['higat_masac', 'random', 'greedy', 'mappo', 'maddpg', 'gnn_ddqn'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for experiment')
    parser.add_argument('--test-env', action='store_true', help='Test environment step logic without training')
    parser.add_argument('--dummy-run', action='store_true', help='Test 2 episodes only for pipeline verification')
    parser.add_argument('--override', nargs='*', default=[], help='Config overrides in format section.key=value, e.g. env.num_vehicles=20')
    parser.add_argument('--out-prefix', type=str, default='', help='Prefix for output result files, e.g. scenario_s2')
    return parser.parse_args()

def apply_overrides(config, overrides):
    """Apply key=value overrides in format 'section.key=value'."""
    for override in overrides:
        key_path, value = override.split('=')
        parts = key_path.split('.')
        d = config
        for part in parts[:-1]:
            d = d[part]
        # Auto-cast to int/float if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string
        d[parts[-1]] = value
    return config

def train(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply any CLI overrides
    if hasattr(args, 'override') and args.override:
        config = apply_overrides(config, args.override)
        
    env = VehicularEnv(config)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.algo == 'higat_masac':
        agent = HiGAT_MASAC(config)
    elif args.algo == 'mappo':
        agent = MAPPOAgent(config)
    elif args.algo == 'maddpg':
        agent = MADDPGAgent(config)
    elif args.algo == 'gnn_ddqn':
        agent = GNNDDQNAgent(config)
    else:
        agent = RandomGreedyBaseline(config)

    episodes = 2 if args.dummy_run else config['rl']['training_episodes']
    t_kd = config['rl']['t_kd_episodes']
    eval_interval = config['rl']['eval_interval']
    
    macro_steps = 2 if args.dummy_run else env.macro_steps_per_episode
    micro_steps = env.micro_steps_per_macro
    
    reward_history = []
    
    # Metrics
    metrics_history = {'delay': [], 'energy': [], 'throughput': []}
    
    for episode in range(episodes):
        macro_state, _ = env.reset(seed=episode)
        ep_reward = 0.0
        
        ep_delay = []
        ep_energy = 0.0
        ep_throughput = []
        
        for t_mac in range(macro_steps):
            # 1. Macro tier execution
            macro_actions = {'subbands': [], 'f_mec_allocated': []}
            
            if args.algo == 'higat_masac':
                subband, fmec = agent.select_macro_action(macro_state)
            elif args.algo in ['mappo', 'maddpg', 'gnn_ddqn']:
                subband, fmec = agent.select_macro_action(macro_state)
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
                
                if args.algo == 'higat_masac':
                    action = agent.select_micro_action(micro_state_list)
                    # action is [p_tx_scalar, alpha_scalar] (N, 2)
                    p_val = action[:, 0]
                    alpha_val = action[:, 1]
                    alpha = (alpha_val + 1) / 2.0
                    chosen_subbands = np.argmax(subband, axis=1)
                    p_tx[np.arange(env.num_vehicles), chosen_subbands] = (p_val + 1) / 2.0 * env.max_tx_power
                else:    
                    # Each vehicle decides its micro action
                    for n in range(env.num_vehicles):
                        if args.algo in ['mappo', 'maddpg', 'gnn_ddqn']:
                            action = agent.select_micro_action(micro_state_list)
                            p_v, a_v = action[n] if len(np.shape(action)) > 1 else action
                            alpha[n] = (a_v + 1) / 2.0 if a_v is not None else 0.5
                            p_tx[n, np.argmax(subband[n])] = (p_v + 1) / 2.0 * env.max_tx_power if p_v is not None else 0.5 * env.max_tx_power
                        else:
                            act = agent.select_micro_action_random(micro_state_list) if args.algo == 'random' else agent.select_micro_action_greedy(micro_state_list)
                            alpha[n] = act[1]
                            p_tx[n, np.argmax(subband[n])] = act[0] * env.max_tx_power
                
                micro_actions = {'p_tx': p_tx, 'alpha': alpha}
                next_micro_state_list, rewards, done, info = env.step_micro(micro_actions, macro_actions)
                
                ep_reward += np.sum(rewards)
                ep_delay.append(np.mean(info['delays']) * 1000) # Convert to ms
                ep_energy += np.sum(info['energies'])
                ep_throughput.append(np.sum(info['throughputs']))
                
                if args.algo == 'higat_masac':
                    # Add to micro replay buffer
                    sg = [t.to(agent.device) if t is not None else None for t in micro_state_list]
                    nsg = [t.to(agent.device) if t is not None else None for t in next_micro_state_list]
                    state_embeds = agent.micro_encoder(*sg).detach().cpu().numpy()
                    next_state_embeds = agent.micro_encoder(*nsg).detach().cpu().numpy()
                    for n in range(env.num_vehicles):
                        agent.micro_buffer.add(
                            state_embeds[n],
                            np.array([p_val[n], alpha_val[n]]),
                            rewards[n],
                            next_state_embeds[n],
                            done
                        )
                    agent.train_micro()
                    
                micro_state_list = next_micro_state_list
                
            # After macro slot, add transition to macro buffer
            if args.algo == 'higat_masac':
                msg = [t.to(agent.device) if t is not None else None for t in macro_state]
                mnsg = [t.to(agent.device) if t is not None else None for t in env._get_macro_states()]
                m_state_embeds = agent.macro_encoder(*msg).detach().cpu().numpy()
                m_next_state = agent.macro_encoder(*mnsg).detach().cpu().numpy()
                
                global_state = np.mean(m_state_embeds, axis=0)
                global_next_state = np.mean(m_next_state, axis=0)
                agent.macro_buffer.add(
                    global_state,
                    subband,
                    fmec,
                    np.sum(rewards)/env.num_vehicles,
                    global_next_state,
                    done
                )
                agent.train_macro()
                macro_state = env._get_macro_states()
        
        if args.algo == 'higat_masac' and episode % t_kd == 0:
            agent.knowledge_distillation()
            
        metrics_history['delay'].append(np.mean(ep_delay))
        metrics_history['energy'].append(ep_energy)
        metrics_history['throughput'].append(np.mean(ep_throughput))
            
        print(f"Episode {episode}: Reward = {ep_reward:.2f}, Delay={np.mean(ep_delay):.2f}ms, Energy={ep_energy:.2f}J, Throughput={np.mean(ep_throughput):.2f}Mbps")
        reward_history.append(ep_reward)
        
        # Save checkpoints or plot periodically...
        if episode > 0 and episode % eval_interval == 0 and args.algo == 'higat_masac':
            os.makedirs('results/higat_masac/plots', exist_ok=True)
            plt.plot(reward_history)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig(f"results/higat_masac/plots/{args.algo}_seed_{args.seed}_reward_curve.png")
            plt.close()
            
    # Save final average metrics (last 3 episodes)
    avg_delay = np.mean(metrics_history['delay'][-3:])
    avg_energy = np.mean(metrics_history['energy'][-3:])
    avg_throughput = np.mean(metrics_history['throughput'][-3:])
    
    os.makedirs('results/higat_masac', exist_ok=True)
    prefix = getattr(args, 'out_prefix', '') or ''
    tag = f"{prefix}_{args.algo}_seed_{args.seed}" if prefix else f"{args.algo}_seed_{args.seed}"
    with open(f'results/higat_masac/{tag}_metrics.json', 'w') as f:
        json.dump({
            'delay': float(avg_delay),
            'energy': float(avg_energy),
            'throughput': float(avg_throughput)
        }, f)
            
    print("Training Complete!")
    
if __name__ == "__main__":
    args = parse_args()
    if args.test_env:
        # Just run environment loops to verify shapes doesn't crash
        args.algo = 'random'
        args.dummy_run = True
    train(args)
