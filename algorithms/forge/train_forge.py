"""
FORGE Training Script

Implements the full FORGE Algorithm (Section 3.4 of algorithms/forge/forge.md):

  Step 1:  OBSERVE  — each agent observes local graph state
  Step 2:  ENCODE   — compute HiGAT macro/micro embeddings
  Step 3-4: MASK + RECONSTRUCT — GMAE forward pass, drift score ξ_t
  Step 5:  BAYESIAN CPD — NBOCD posterior update
  Step 6:  CLASSIFY DRIFT — no_drift / soft_drift / hard_drift / structural_drift
  Step 7:  ACT      — macro + micro actors sample actions
  Step 8:  EVOLVE REWARD — ERPG updates α/β/γ weights
  Step 9:  STORE    — push transitions to local replay buffers
  Step 10: UPDATE POLICY — SAC actor/critic/alpha update
  Step 11: UPDATE GMAE   — fine-tune GMAE + TCN on drift events
  Step 12: FEDERATED AGGREGATION — HFA sync on hard/structural drift

Run from repo root:
  python -m algorithms.forge.train_forge --config common/configs/default.yaml --episodes 500
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
from algorithms.forge.models.hfa import HFA


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FORGE Training")
    p.add_argument('--config',    type=str, default='common/configs/default.yaml')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--episodes',  type=int, default=500, help='Training episodes')
    p.add_argument('--warmup',    type=int, default=200, help='GMAE warm-up steps')
    p.add_argument('--save-dir',  type=str, default='results/forge')
    p.add_argument('--eval-freq', type=int, default=10,  help='Eval every N episodes')
    return p.parse_args()


# ---------------------------------------------------------------------------
# GMAE warm-up: pretrain on initial graphs before RL starts
# ---------------------------------------------------------------------------
def warmup_gmae(agents, env, warmup_steps, seed):
    """Pre-train GMAE representations on the initial graph distribution."""
    print(f"[GMAE Warm-up] Running {warmup_steps} steps...")
    mac_state, _ = env.reset(seed=seed)

    for step in range(warmup_steps):
        for agent in agents:
            x, ei, _ = agent.pack_macro_graph(mac_state)
            loss_gmae, xi, _ = agent.gmae_cpd(x, ei)
            agent.gmae_optimizer.zero_grad()
            if loss_gmae.requires_grad:
                loss_gmae.backward()
                agent.gmae_optimizer.step()

        # Step env with random actions to gather diverse graphs
        n_veh = env.num_vehicles
        K     = env.K
        dummy_macro = {
            'subbands':        np.eye(K)[np.random.choice(K, n_veh)],
            'f_mec_allocated': np.random.uniform(0.1, 1.0, n_veh)
        }
        dummy_micro = {
            'p_tx':  np.random.uniform(0, env.max_tx_power, (n_veh, K)),
            'alpha': np.random.uniform(0, 1, n_veh)
        }
        _, _, _, _ = env.step_micro(dummy_micro, dummy_macro)
        env.mobility_model.step()

        if (step + 1) % 50 == 0:
            print(f"  Warm-up step {step + 1}/{warmup_steps}")

    print("[GMAE Warm-up] Complete.\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = VehicularEnv(config)
    num_rsus    = env.num_rsus
    num_vehicles = env.num_vehicles
    K = env.K

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'),      exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)

    # --- Initialize M Federated FORGE Agents ---
    agents = [FORGE_SAC(config) for _ in range(num_rsus)]
    hfa    = HFA(num_rsus)
    print(f"Initialized {num_rsus} Federated FORGE Nodes on device: {agents[0].device}")

    # --- GMAE Warm-Up (Initialize: Section 3.4) ---
    warmup_gmae(agents, env, args.warmup, args.seed)

    # Tracking
    return_history      = []
    drift_event_history = []
    global_hfa_rounds   = 0

    # ---------------------------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------------------------
    for episode in range(args.episodes):
        # Reset environment and NBOCD posteriors
        mac_state, _ = env.reset(seed=args.seed + episode)
        for agent in agents:
            agent.nbocd.reset()

        episode_return  = 0.0
        episode_drifts  = 0
        ep_delays       = []
        ep_energies     = []
        ep_throughputs  = []

        # ── MACRO LOOP ──────────────────────────────────────────────────────
        for t in range(env.macro_steps_per_episode):

            drift_flags       = []
            macro_subband_all = np.zeros((num_vehicles, K))
            macro_fmec_all    = np.zeros(num_vehicles)
            macro_states_emb  = []   # one embedding per RSU

            curr_loss_gmae_per_agent = []
            curr_loss_tcn_per_agent  = []

            for m, agent in enumerate(agents):
                local_graph = [mac_state[m]]

                # ── Step 1: OBSERVE / Step 2: ENCODE ────────────────────────
                x, ei, batch = agent.pack_macro_graph(local_graph)
                with torch.no_grad():
                    macro_emb = agent.macro_encoder(x, ei, batch=batch)
                    if macro_emb.shape[0] > 1:
                        macro_emb = macro_emb.mean(0, keepdim=True)
                macro_states_emb.append(macro_emb)

                # ── Steps 3-5: DRIFT DETECTION ──────────────────────────────
                status, loss_gmae, loss_tcn = agent.detect_drift(local_graph)
                drift_flags.append(status)
                curr_loss_gmae_per_agent.append(loss_gmae)
                curr_loss_tcn_per_agent.append(loss_tcn)

                if status in ("soft_drift", "hard_drift", "structural_drift"):
                    episode_drifts += 1

                # ── Step 7: ACT (Macro) ─────────────────────────────────────
                subband_np, fmec_np = agent.select_macro_action(local_graph)

                # Map subband/fmec to vehicles in this cluster
                cluster_vehs = mac_state[m]['cluster_vehicles']
                if len(cluster_vehs) > 0 and subband_np.ndim >= 2:
                    macro_subband_all[cluster_vehs] = subband_np[: len(cluster_vehs)]
                    macro_fmec_all[cluster_vehs]    = fmec_np[: len(cluster_vehs)]

            macro_actions_dict = {
                'subbands':        macro_subband_all,
                'f_mec_allocated': macro_fmec_all
            }

            # ── MICRO LOOP ───────────────────────────────────────────────────
            micro_states = env.get_micro_state(macro_actions_dict)

            macro_prev_emb = {}
            for m, agent in enumerate(agents):
                macro_prev_emb[m] = macro_states_emb[m]

            for micro_step in range(env.micro_steps_per_macro):
                # ── Step 7: ACT (Micro) ─────────────────────────────────────
                micro_actions_raw = agents[0].select_micro_action(micro_states)  # shared micro policy

                # Map [-1,1] action to physical actions
                p_tx  = ((micro_actions_raw[:, 0] + 1.0) / 2.0) * env.max_tx_power  # [0, P_max]
                alpha = (micro_actions_raw[:, 1] + 1.0) / 2.0                       # [0, 1]

                micro_actions = {
                    'p_tx':  np.tile(p_tx[:, None], (1, K)) / K,  # spread evenly across subbands
                    'alpha': alpha
                }

                next_micro_states, rewards, _, info = env.step_micro(micro_actions, macro_actions_dict)

                ep_delays.extend(info['delays'])
                ep_energies.extend(info['energies'])
                ep_throughputs.extend(info['throughputs'])

                avg_delay      = float(np.mean(info['delays']))
                avg_energy     = float(np.mean(info['energies']))
                avg_throughput = float(np.mean(info['throughputs']))

                # ── Step 8: EVOLVE REWARD (ERPG) ────────────────────────────
                for agent in agents:
                    agent.erpg.update_history(avg_delay, avg_energy, avg_throughput)
                    dw = agent.erpg.evolve_weights(agent.device)

                # Use agent 0's weights as canonical (they converge to similar values in practice)
                dynamic_weights = agents[0].erpg.current_weights
                adapted_reward  = agents[0].erpg.get_reward(avg_delay, avg_energy, avg_throughput)
                episode_return += adapted_reward

                # ── Step 9: STORE ────────────────────────────────────────────
                # Store a single macro embedding + average micro reward per agent
                for m, agent in enumerate(agents):
                    macro_emb_np = macro_prev_emb[m].detach().cpu().numpy().squeeze(0)  # [embed_dim]

                    # Macro buffer: store (s, subband, fmec, r, s', done) with macro embeddings
                    cluster_vehs = mac_state[m]['cluster_vehicles']
                    if len(cluster_vehs) > 0:
                        sub_m  = macro_subband_all[cluster_vehs[0]]      # [K]
                        sub_m  = np.eye(K)[np.argmax(sub_m)]            # one-hot [K]
                        fmec_m = macro_fmec_all[cluster_vehs].mean()

                        # Pad to fixed num_vehicles size
                        sub_matrix  = np.zeros((num_vehicles, K))
                        fmec_vec    = np.zeros(num_vehicles)
                        for i, v in enumerate(cluster_vehs[:num_vehicles]):
                            sub_matrix[i]  = sub_m
                            fmec_vec[i]    = fmec_m

                        r_cluster = float(np.mean(rewards[cluster_vehs]) if len(cluster_vehs) > 0 else adapted_reward)
                        agent.macro_buffer.add(macro_emb_np, sub_matrix, fmec_vec,
                                               r_cluster, macro_emb_np, float(micro_step == env.micro_steps_per_macro - 1))

                    # Micro buffer: store per-vehicle transitions
                    for n in range(num_vehicles):
                        x_micro, ei_micro = agent.pack_micro_graph([micro_states[n]])
                        with torch.no_grad():
                            micro_emb = agent.micro_encoder(x_micro, ei_micro)
                        micro_emb_np = micro_emb[0].cpu().numpy()  # [embed_dim]

                        x_micro_next, ei_next = agent.pack_micro_graph([next_micro_states[n]])
                        with torch.no_grad():
                            micro_emb_next = agent.micro_encoder(x_micro_next, ei_next)
                        micro_emb_next_np = micro_emb_next[0].cpu().numpy()

                        agent.micro_buffer.add(micro_emb_np, micro_actions_raw[n],
                                               adapted_reward, micro_emb_next_np,
                                               float(micro_step == env.micro_steps_per_macro - 1))

                micro_states = next_micro_states

                # ── Step 10: UPDATE POLICY ──────────────────────────────────
                for agent in agents:
                    agent.train_macro(dynamic_reward_weights=dynamic_weights)
                    agent.train_micro(dynamic_reward_weights=dynamic_weights)

            # ── Step 11: UPDATE GMAE + TCN ──────────────────────────────────
            for m, agent in enumerate(agents):
                status = drift_flags[m]
                if status in ("soft_drift", "hard_drift"):
                    # Combined loss: L_total = L_GMAE^current + λ_mem * L_GMAE^replay
                    agent.fine_tune_gmae(curr_loss_gmae_per_agent[m])
                else:
                    # Still do a plain update on current loss to keep GMAE fresh
                    l = curr_loss_gmae_per_agent[m]
                    if l.requires_grad:
                        agent.gmae_optimizer.zero_grad()
                        l.backward()
                        agent.gmae_optimizer.step()

                # Update TCN
                agent.update_tcn(curr_loss_tcn_per_agent[m])

            # ── Step 12: FEDERATED AGGREGATION ──────────────────────────────
            any_hard = any(f in ("hard_drift", "structural_drift") for f in drift_flags)
            if any_hard:
                global_hfa_rounds += 1

                macro_dicts = [a.get_state_dict()['macro_actor'] for a in agents]
                micro_dicts = [a.get_state_dict()['micro_actor']  for a in agents]

                avg_macro = hfa.aggregate_macro(macro_dicts)
                avg_micro = hfa.aggregate_micro(micro_dicts)

                for a in agents:
                    sd = a.get_state_dict()
                    sd['macro_actor'] = avg_macro
                    sd['micro_actor'] = avg_micro
                    a.load_state_dict(sd)

                # On structural drift: also reset NBOCD run-length posteriors
                if "structural_drift" in drift_flags:
                    for a in agents:
                        a.nbocd.reset()

            # Advance macro env
            env.step_macro(macro_actions_dict)

            # Update local mac_state for next macro step
            mac_state = env._get_macro_state()

        return_history.append(episode_return)
        drift_event_history.append(episode_drifts)

        if (episode + 1) % args.eval_freq == 0:
            avg_tp    = float(np.mean(ep_throughputs)) if ep_throughputs else 0.0
            avg_lat   = float(np.mean(ep_delays))      if ep_delays      else 0.0
            avg_nrg   = float(np.mean(ep_energies))    if ep_energies    else 0.0
            print(f"Ep [{episode+1:4d}/{args.episodes}] | "
                  f"Ret={episode_return:8.2f} | "
                  f"Tp={avg_tp:.3f} Mbps | "
                  f"Lat={avg_lat*1e3:.2f} ms | "
                  f"Drifts={episode_drifts} | "
                  f"HFA={global_hfa_rounds}")

    # ---------------------------------------------------------------------------
    # Save results and plots
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(return_history, label='Episode Return')
    plt.xlabel('Episode')
    plt.ylabel('ERPG-Weighted Return')
    plt.title('FORGE Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', f'forge_seed_{args.seed}_reward_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(drift_event_history, color='orange', label='Drift Events/Episode')
    plt.xlabel('Episode')
    plt.ylabel('Drift Events')
    plt.title('FORGE Drift Detection History')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', f'forge_seed_{args.seed}_drift_history.png'))
    plt.close()

    metrics = {
        'final_avg_return':   float(np.mean(return_history[-10:])),
        'total_hfa_rounds':   global_hfa_rounds,
        'total_drift_events': int(np.sum(drift_event_history)),
        'seed':               args.seed,
        'episodes':           args.episodes,
    }
    with open(os.path.join(args.save_dir, f'forge_seed_{args.seed}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining finished.")
    print(f"  Final avg return (last 10): {metrics['final_avg_return']:.2f}")
    print(f"  Total HFA rounds triggered: {global_hfa_rounds}")
    print(f"  Results saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
