"""
HiGAT-MASAC Training Script (updated for dict-based env API)
"""

import yaml
import argparse
import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt

from common.env.vehicular_env import VehicularEnv
from algorithms.higat_masac.models.gat_encoder import GATEncoder
from algorithms.higat_masac.models.sac_micro import MicroActor, MicroCritic
from algorithms.higat_masac.models.sac_macro import MacroActor, MacroCritic
from algorithms.higat_masac.models.higat_masac import ReplayBuffer, soft_update
from common.baselines.random_greedy import RandomGreedyBaseline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='common/configs/default.yaml')
    parser.add_argument('--algo', type=str, default='higat_masac',
                        choices=['higat_masac', 'random', 'greedy'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dummy-run', action='store_true')
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--out-prefix', type=str, default='')
    return parser.parse_args()


def apply_overrides(config, overrides):
    for override in overrides:
        key_path, value = override.split('=')
        parts = key_path.split('.')
        d = config
        for part in parts[:-1]:
            d = d[part]
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        d[parts[-1]] = value
    return config


def pack_macro_graph(mac_state, device):
    """Convert list-of-RSU-dicts → (x, edge_index, batch) tensors."""
    all_x, batch = [], []
    for d in mac_state:
        feats = d['node_features']
        if len(feats) == 0:
            continue
        all_x.append(feats)
        batch.extend([d['rsu_id']] * len(feats))
    if not all_x:
        x = torch.zeros((1, 5), dtype=torch.float32, device=device)
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        bt = torch.zeros(1, dtype=torch.long, device=device)
        return x, ei, bt
    x = torch.tensor(np.vstack(all_x), dtype=torch.float32, device=device)
    edge_index, offset = [], 0
    for d in mac_state:
        N = len(d['node_features'])
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index.append([i + offset, j + offset])
        offset += N
    if edge_index:
        ei = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    else:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
    bt = torch.tensor(batch, dtype=torch.long, device=device)
    return x, ei, bt


def pack_micro_graph(micro_states, device):
    """Convert list of vehicle micro state dicts → (x, edge_index)."""
    all_x = [s['local_features'] for s in micro_states]
    x = torch.tensor(np.array(all_x), dtype=torch.float32, device=device)
    N = x.shape[0]
    edge_index = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
    if edge_index:
        ei = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    else:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
    return x, ei


class HiGAT_MASAC_Trainer:
    """Self-contained HiGAT-MASAC trainer using the dict-based env."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config['rl']['gamma']
        self.tau = config['rl']['tau']
        lr_a = config['rl']['actor_lr']
        lr_c = config['rl']['critic_lr']
        self.batch_size = config['rl']['batch_size']
        N = config['env']['num_vehicles']
        K = config['env']['num_subbands']
        F = config['env']['rsu_mec_capacity_ghz']
        D = config['rl']['gat_embed_dim']

        self.macro_encoder = GATEncoder(5, config).to(self.device)
        self.micro_encoder  = GATEncoder(4, config).to(self.device)
        self.macro_actor  = MacroActor(D, N, K, F).to(self.device)
        self.macro_critic = MacroCritic(D, N, K).to(self.device)
        self.macro_critic_t = MacroCritic(D, N, K).to(self.device)
        self.macro_critic_t.load_state_dict(self.macro_critic.state_dict())
        self.micro_actor  = MicroActor(D, 2, 1.0).to(self.device)
        self.micro_critic = MicroCritic(D, 2).to(self.device)
        self.micro_critic_t = MicroCritic(D, 2).to(self.device)
        self.micro_critic_t.load_state_dict(self.micro_critic.state_dict())

        import torch.optim as optim, torch.nn.functional as F_func
        self.F = F_func
        self.macro_a_opt = optim.Adam(self.macro_actor.parameters(), lr=lr_a)
        self.macro_c_opt = optim.Adam(self.macro_critic.parameters(), lr=lr_c)
        self.micro_a_opt = optim.Adam(self.micro_actor.parameters(), lr=lr_a)
        self.micro_c_opt = optim.Adam(self.micro_critic.parameters(), lr=lr_c)

        self.log_alpha_mac = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_mac_opt = optim.Adam([self.log_alpha_mac], lr=lr_a)
        self.tgt_ent_mac   = -float(N * K + N)
        self.log_alpha_mic = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_mic_opt = optim.Adam([self.log_alpha_mic], lr=lr_a)
        self.tgt_ent_mic   = -2.0

        self.macro_buf = ReplayBuffer(config['rl']['buffer_size'], D,
                                       is_macro=True, num_vehicles=N, num_subbands=K)
        self.micro_buf = ReplayBuffer(config['rl']['buffer_size'], D, action_dim=2)
        self.N, self.K, self.D = N, K, D

    def select_macro_action(self, mac_state):
        x, ei, bt = pack_macro_graph(mac_state, self.device)
        with torch.no_grad():
            emb = self.macro_encoder(x, ei, batch=bt)
            if emb.shape[0] > 1:
                emb = emb.mean(0, keepdim=True)
            subband, fmec_norm, _, _ = self.macro_actor.sample(emb)
        subband_np = subband.cpu().numpy().squeeze(0)  # [N, K]
        # Scale fmec from sigmoid [0,1] to actual Hz
        fmec_np = fmec_norm.cpu().numpy().squeeze(0) * self.config['env']['rsu_mec_capacity_ghz'] * 1e9
        fmec_np = np.clip(fmec_np, 1e8, self.config['env']['rsu_mec_capacity_ghz'] * 1e9)  # min 0.1 GHz
        return subband_np, fmec_np

    def select_micro_action(self, micro_states):
        x, ei = pack_micro_graph(micro_states, self.device)
        with torch.no_grad():
            emb = self.micro_encoder(x, ei)
            action, _, _ = self.micro_actor.sample(emb)
        return action.cpu().numpy()

    def embed_macro(self, mac_state):
        x, ei, bt = pack_macro_graph(mac_state, self.device)
        with torch.no_grad():
            emb = self.macro_encoder(x, ei, batch=bt)
            return emb.mean(0).cpu().numpy()

    def embed_micro(self, micro_states):
        x, ei = pack_micro_graph(micro_states, self.device)
        with torch.no_grad():
            return self.micro_encoder(x, ei).cpu().numpy()

    def train_macro(self):
        if self.macro_buf.size < self.batch_size:
            return
        import torch.nn.functional as F_func
        state, subband, fmec, reward, next_state, done = self.macro_buf.sample(self.batch_size)
        state, subband, fmec, reward, next_state, done = [t.to(self.device) for t in
                                                           [state, subband, fmec, reward, next_state, done]]
        alpha = self.log_alpha_mac.exp().detach()
        with torch.no_grad():
            ns, nf, nlp, _ = self.macro_actor.sample(next_state)
            tQ1, tQ2 = self.macro_critic_t(next_state, ns, nf)
            tQ = reward + (1 - done) * self.gamma * (torch.min(tQ1, tQ2) - alpha * nlp)
        cQ1, cQ2 = self.macro_critic(state, subband, fmec)
        cl = F_func.mse_loss(cQ1, tQ) + F_func.mse_loss(cQ2, tQ)
        self.macro_c_opt.zero_grad(); cl.backward()
        torch.nn.utils.clip_grad_norm_(self.macro_critic.parameters(), 1.0)
        self.macro_c_opt.step()
        ns2, nf2, lp2, _ = self.macro_actor.sample(state)
        aQ1, aQ2 = self.macro_critic(state, ns2, nf2)
        al = (alpha * lp2 - torch.min(aQ1, aQ2)).mean()
        self.macro_a_opt.zero_grad(); al.backward()
        torch.nn.utils.clip_grad_norm_(self.macro_actor.parameters(), 1.0)
        self.macro_a_opt.step()
        ent_l = (self.log_alpha_mac * (-lp2.detach() - self.tgt_ent_mac)).mean()
        self.alpha_mac_opt.zero_grad(); ent_l.backward(); self.alpha_mac_opt.step()
        soft_update(self.macro_critic_t, self.macro_critic, self.tau)

    def train_micro(self):
        if self.micro_buf.size < self.batch_size:
            return
        import torch.nn.functional as F_func
        state, action, reward, next_state, done = self.micro_buf.sample(self.batch_size)
        state, action, reward, next_state, done = [t.to(self.device) for t in
                                                    [state, action, reward, next_state, done]]
        alpha = self.log_alpha_mic.exp().detach()
        with torch.no_grad():
            na, nlp, _ = self.micro_actor.sample(next_state)
            tQ1, tQ2 = self.micro_critic_t(next_state, na)
            tQ = reward + (1 - done) * self.gamma * (torch.min(tQ1, tQ2) - alpha * nlp)
        cQ1, cQ2 = self.micro_critic(state, action)
        cl = F_func.mse_loss(cQ1, tQ) + F_func.mse_loss(cQ2, tQ)
        self.micro_c_opt.zero_grad(); cl.backward()
        torch.nn.utils.clip_grad_norm_(self.micro_critic.parameters(), 1.0)
        self.micro_c_opt.step()
        na2, lp2, _ = self.micro_actor.sample(state)
        aQ1, aQ2 = self.micro_critic(state, na2)
        al = (alpha * lp2 - torch.min(aQ1, aQ2)).mean()
        self.micro_a_opt.zero_grad(); al.backward()
        torch.nn.utils.clip_grad_norm_(self.micro_actor.parameters(), 1.0)
        self.micro_a_opt.step()
        ent_l = (self.log_alpha_mic * (-lp2.detach() - self.tgt_ent_mic)).mean()
        self.alpha_mic_opt.zero_grad(); ent_l.backward(); self.alpha_mic_opt.step()
        soft_update(self.micro_critic_t, self.micro_critic, self.tau)

    def knowledge_distillation(self):
        for tc, sc in zip(self.micro_encoder.convs[1:], self.macro_encoder.convs[1:]):
            soft_update(tc, sc, tau=0.5)


def train(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.override:
        config = apply_overrides(config, args.override)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = VehicularEnv(config)
    N = env.num_vehicles
    K = env.K

    if args.algo == 'higat_masac':
        agent = HiGAT_MASAC_Trainer(config)
    else:
        agent = RandomGreedyBaseline(config)

    episodes  = 2 if args.dummy_run else config['rl']['training_episodes']
    t_kd      = config['rl']['t_kd_episodes']
    macro_steps = 2 if args.dummy_run else env.macro_steps_per_episode
    micro_steps = env.micro_steps_per_macro

    reward_history, delay_history, tp_history = [], [], []

    for episode in range(episodes):
        mac_state, _ = env.reset(seed=episode)
        ep_reward = 0.0
        ep_delay, ep_energy, ep_tp = [], 0.0, []

        for t_mac in range(macro_steps):
            # ── Macro action selection ──────────────────────────────────────
            if args.algo == 'higat_masac':
                subband, fmec = agent.select_macro_action(mac_state)
            elif args.algo == 'random':
                subband = np.eye(K)[np.random.choice(K, N)]
                fmec    = np.random.uniform(0.1, config['env']['rsu_mec_capacity_ghz'], N)
            else:  # greedy
                subband = np.eye(K)[np.arange(N) % K]
                fmec    = np.ones(N) * config['env']['rsu_mec_capacity_ghz'] * 0.8

            macro_actions = {'subbands': subband, 'f_mec_allocated': fmec}
            micro_state_list = env.get_micro_state(macro_actions)

            prev_macro_emb = agent.embed_macro(mac_state) if args.algo == 'higat_masac' else None

            # ── Micro loop ──────────────────────────────────────────────────
            for t_mic in range(micro_steps):
                if args.algo == 'higat_masac':
                    raw_action = agent.select_micro_action(micro_state_list)
                    p_val   = raw_action[:, 0]
                    a_val   = raw_action[:, 1]
                    # Map tanh [-1,1] to [0.05, 0.95] to avoid zero-rate issues
                    alpha   = np.clip((a_val + 1) / 2.0, 0.05, 0.95)
                    p_tx    = np.zeros((N, K))
                    chosen_k = np.argmax(subband, axis=1)
                    p_tx[np.arange(N), chosen_k] = np.clip((p_val + 1) / 2.0, 0.01, 1.0) * env.max_tx_power
                else:
                    alpha = np.ones(N) * 0.7
                    p_tx  = np.zeros((N, K))
                    for n in range(N):
                        p_tx[n, np.argmax(subband[n])] = env.max_tx_power * 0.7

                micro_actions = {'p_tx': p_tx, 'alpha': alpha}
                next_micro_states, rewards, done, info = env.step_micro(micro_actions, macro_actions)

                ep_reward += float(np.mean(rewards))  # per-vehicle per-step average
                ep_delay.append(float(np.mean(info['delays']) * 1e3))
                ep_energy += float(np.mean(info['energies']))
                ep_tp.append(float(np.mean(info['throughputs'])))

                if args.algo == 'higat_masac':
                    prev_micro_emb = agent.embed_micro(micro_state_list)
                    next_micro_emb = agent.embed_micro(next_micro_states)
                    for n in range(N):
                        agent.micro_buf.add(prev_micro_emb[n],
                                            np.array([p_val[n], a_val[n]]),
                                            rewards[n],
                                            next_micro_emb[n],
                                            float(done))
                    agent.train_micro()

                micro_state_list = next_micro_states

            # ── Macro buffer update ─────────────────────────────────────────
            if args.algo == 'higat_masac':
                next_mac_state = env._get_macro_state()
                next_macro_emb = agent.embed_macro(next_mac_state)
                agent.macro_buf.add(prev_macro_emb, subband, fmec,
                                    np.mean(rewards), next_macro_emb, float(done))
                agent.train_macro()
                mac_state = next_mac_state
            else:
                mac_state = env._get_macro_state()

        if args.algo == 'higat_masac' and episode % t_kd == 0:
            agent.knowledge_distillation()

        reward_history.append(ep_reward)
        delay_history.append(np.mean(ep_delay))
        tp_history.append(np.mean(ep_tp))

        if (episode + 1) % 10 == 0:
            print(f"Ep [{episode+1:4d}/{episodes}] Reward={ep_reward:.2f}  "
                  f"Delay={np.mean(ep_delay):.2f}ms  Tp={np.mean(ep_tp):.2f}Mbps")

    # ── Save results ────────────────────────────────────────────────────────
    os.makedirs('results/higat_masac/plots', exist_ok=True)
    plt.plot(reward_history)
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title(f"HiGAT-MASAC ({args.algo}) Training")
    plt.tight_layout()
    plt.savefig(f"results/higat_masac/plots/{args.algo}_seed_{args.seed}_reward_curve.png")
    plt.close()

    prefix = getattr(args, 'out_prefix', '') or ''
    tag = f"{prefix}_{args.algo}_seed_{args.seed}" if prefix else f"{args.algo}_seed_{args.seed}"
    with open(f'results/higat_masac/{tag}_metrics.json', 'w') as f:
        json.dump({
            'delay_ms':     float(np.mean(delay_history[-10:])),
            'throughput_Mbps': float(np.mean(tp_history[-10:])),
            'final_reward': float(np.mean(reward_history[-10:])),
            'seed': args.seed, 'algo': args.algo
        }, f, indent=2)

    print(f"\n=== Final (last 10 episodes) ===")
    print(f"  Avg Delay:      {np.mean(delay_history[-10:]):.2f} ms")
    print(f"  Avg Throughput: {np.mean(tp_history[-10:]):.2f} Mbps")
    print(f"  Avg Reward:     {np.mean(reward_history[-10:]):.2f}")
    print("Training Complete!")
    return {
        'delay_ms': float(np.mean(delay_history[-10:])),
        'throughput_Mbps': float(np.mean(tp_history[-10:])),
        'final_reward': float(np.mean(reward_history[-10:]))
    }


if __name__ == "__main__":
    args = parse_args()
    train(args)
