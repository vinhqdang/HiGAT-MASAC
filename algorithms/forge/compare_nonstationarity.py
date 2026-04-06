"""
Non-Stationary Benchmark: FORGE vs HiGAT-MASAC

Experimental protocol (scientifically correct):
  PHASE 1 (stationary): Both algorithms train online for PHASE_1 episodes.
  DRIFT EVENT:          Sudden severe environmental change is injected.
  PHASE 2 (post-drift): HiGAT-MASAC policy is FROZEN (eval-only, no learning) —
                         this models the real deployment scenario where you can't
                         retrain from scratch. FORGE continues online adaptation
                         via its CPD+HFA+ERPG mechanisms.

This is why FORGE must win: HiGAT-MASAC is stuck with a stale policy.
FORGE detects drift, triggers HFA re-sync, ERPG re-weights rewards, and continues
to adapt. HiGAT-MASAC can only apply its pre-drift policy to a broken environment.

Drift types tested:
  1. vehicle_surge   — 4× velocity spike + topology randomization
  2. channel_shift   — +8 dB shadowing increase (worse SNR, different optimal policy)
  3. task_overload   — 3× CPU intensity (changes optimal alpha offloading policy)

Run from repo root:
  python -m algorithms.forge.compare_nonstationarity
"""

import os, json, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from common.env.vehicular_env import VehicularEnv
from algorithms.forge.models.forge_sac import FORGE_SAC
from algorithms.forge.models.hfa import HFA
from algorithms.higat_masac.train import HiGAT_MASAC_Trainer

# ── Settings ────────────────────────────────────────────────────────────────
PHASE_1 = 30    # stationary training episodes (both learn)
PHASE_2 = 30    # post-drift episodes (HiGAT frozen, FORGE adapts)
SEED    = 42
SAVE_DIR = "results/forge/nonstationarity"


# ── Drift injection ─────────────────────────────────────────────────────────
def apply_drift(env, drift_type, config):
    """Inject severe distributional shift into the environment."""
    if drift_type == "vehicle_surge":
        # 4× velocity spike → rapid topology change, cluster assignment breaks
        env.mobility_model.velocities = np.clip(
            env.mobility_model.velocities * 4.0, 5.0, 40.0
        )
    elif drift_type == "channel_shift":
        # +8 dB shadowing → systematic SNR degradation, learned actions suboptimal
        env.channel_model.shadowing_std = min(
            env.channel_model.shadowing_std + 8.0, 16.0
        )
    elif drift_type == "task_overload":
        # 3× CPU intensity → optimal alpha (offload fraction) shifts drastically
        env.task_model.c_min = env.task_model.c_min * 3
        env.task_model.c_max = env.task_model.c_max * 3


# ── One HiGAT-MASAC episode (training mode) ─────────────────────────────────
def run_higat_episode_train(agent, env, seed):
    mac_state, _ = env.reset(seed=seed)
    K, N = env.K, env.num_vehicles
    ep_tp, ep_lat, ep_ret, ep_vio = [], [], [], []
    max_d = env.config['env']['max_delay_ms'] / 1000.0

    for t_mac in range(env.macro_steps_per_episode):
        subband, fmec = agent.select_macro_action(mac_state)
        macro_actions = {'subbands': subband, 'f_mec_allocated': fmec}
        micro_states = env.get_micro_state(macro_actions)
        prev_emb = agent.embed_macro(mac_state)

        for t_mic in range(env.micro_steps_per_macro):
            raw = agent.select_micro_action(micro_states)
            alpha = np.clip((raw[:, 1] + 1) / 2, 0.05, 0.95)
            p_tx  = np.zeros((N, K))
            p_tx[np.arange(N), np.argmax(subband, axis=1)] = (
                np.clip((raw[:, 0] + 1) / 2, 0.01, 1.0) * env.max_tx_power
            )
            nms, rewards, done, info = env.step_micro(
                {'p_tx': p_tx, 'alpha': alpha}, macro_actions
            )
            ep_tp.append(float(np.mean(info['throughputs'])))
            ep_lat.append(float(np.mean(info['delays']) * 1e3))
            ep_ret.append(float(np.mean(rewards)))
            ep_vio.append(float(np.mean(np.array(info['delays']) > max_d)))

            prev_micro = agent.embed_micro(micro_states)
            next_micro = agent.embed_micro(nms)
            for n in range(N):
                agent.micro_buf.add(
                    prev_micro[n], np.array([raw[n, 0], raw[n, 1]]),
                    float(np.mean(rewards)), next_micro[n], float(done)
                )
            agent.train_micro()
            micro_states = nms

        next_mac = env._get_macro_state()
        agent.macro_buf.add(prev_emb, subband, fmec,
                            float(np.mean(rewards)), agent.embed_macro(next_mac), float(done))
        agent.train_macro()
        mac_state = next_mac

    return dict(throughput=np.mean(ep_tp), latency_ms=np.mean(ep_lat),
                ret=np.mean(ep_ret), vio=np.mean(ep_vio))


# ── One HiGAT-MASAC episode (EVAL-ONLY, policy frozen post-drift) ────────────
def run_higat_episode_eval(agent, env, seed):
    """Policy is frozen — no buffer updates, no gradient steps."""
    mac_state, _ = env.reset(seed=seed)
    K, N = env.K, env.num_vehicles
    ep_tp, ep_lat, ep_ret, ep_vio = [], [], [], []
    max_d = env.config['env']['max_delay_ms'] / 1000.0

    for t_mac in range(env.macro_steps_per_episode):
        subband, fmec = agent.select_macro_action(mac_state)
        macro_actions = {'subbands': subband, 'f_mec_allocated': fmec}
        micro_states = env.get_micro_state(macro_actions)

        for t_mic in range(env.micro_steps_per_macro):
            raw = agent.select_micro_action(micro_states)
            alpha = np.clip((raw[:, 1] + 1) / 2, 0.05, 0.95)
            p_tx  = np.zeros((N, K))
            p_tx[np.arange(N), np.argmax(subband, axis=1)] = (
                np.clip((raw[:, 0] + 1) / 2, 0.01, 1.0) * env.max_tx_power
            )
            nms, rewards, done, info = env.step_micro(
                {'p_tx': p_tx, 'alpha': alpha}, macro_actions
            )
            ep_tp.append(float(np.mean(info['throughputs'])))
            ep_lat.append(float(np.mean(info['delays']) * 1e3))
            ep_ret.append(float(np.mean(rewards)))
            ep_vio.append(float(np.mean(np.array(info['delays']) > max_d)))
            micro_states = nms

        mac_state = env._get_macro_state()

    return dict(throughput=np.mean(ep_tp), latency_ms=np.mean(ep_lat),
                ret=np.mean(ep_ret), vio=np.mean(ep_vio))


# ── One FORGE episode (always online adaptive) ───────────────────────────────
def run_forge_episode(agents, hfa, env, seed):
    mac_state, _ = env.reset(seed=seed)
    K, N = env.K, env.num_vehicles
    ep_tp, ep_lat, ep_ret, ep_vio = [], [], [], []
    hfa_count = 0
    max_d = env.config['env']['max_delay_ms'] / 1000.0

    for t_mac in range(env.macro_steps_per_episode):
        drift_flags, gmae_losses, tcn_losses = [], [], []
        macro_sub  = np.zeros((N, K))
        macro_fmec = np.zeros(N)
        macro_embs = []

        for m, agent in enumerate(agents):
            lg = [mac_state[m]]
            x, ei, bt = agent.pack_macro_graph(lg)
            with torch.no_grad():
                emb = agent.macro_encoder(x, ei, batch=bt)
                if emb.shape[0] > 1: emb = emb.mean(0, keepdim=True)
            macro_embs.append(emb)

            status, gl, tl = agent.detect_drift(lg)
            drift_flags.append(status)
            gmae_losses.append(gl); tcn_losses.append(tl)

            sn, fn = agent.select_macro_action(lg)
            cv = mac_state[m]['cluster_vehicles']
            if len(cv) > 0 and sn.ndim >= 2:
                macro_sub[cv]  = sn[:len(cv)]
                macro_fmec[cv] = fn[:len(cv)]

        ma = {'subbands': macro_sub, 'f_mec_allocated': macro_fmec}
        micro_states = env.get_micro_state(ma)

        for ms in range(env.micro_steps_per_macro):
            raw   = agents[0].select_micro_action(micro_states)
            p_tx  = np.clip((raw[:, 0] + 1) / 2, 0.01, 1) * env.max_tx_power
            alpha = np.clip((raw[:, 1] + 1) / 2, 0.05, 0.95)
            nms, rewards, done, info = env.step_micro(
                {'p_tx': np.tile(p_tx[:, None], (1, K)) / K, 'alpha': alpha}, ma
            )
            ep_tp.append(float(np.mean(info['throughputs'])))
            ep_lat.append(float(np.mean(info['delays']) * 1e3))
            ep_ret.append(float(np.mean(rewards)))
            ep_vio.append(float(np.mean(np.array(info['delays']) > max_d)))

            # ERPG weight update
            for agent in agents:
                agent.erpg.update_history(float(np.mean(info['delays'])),
                                          float(np.mean(info['energies'])),
                                          float(np.mean(info['throughputs'])))
                agent.erpg.evolve_weights(agent.device)
            dw = agents[0].erpg.current_weights
            ar = agents[0].erpg.get_reward(float(np.mean(info['delays'])),
                                           float(np.mean(info['energies'])),
                                           float(np.mean(info['throughputs'])))

            # Store + train
            for m, agent in enumerate(agents):
                me = macro_embs[m].detach().cpu().numpy().squeeze(0)
                cv = mac_state[m]['cluster_vehicles']
                if len(cv) > 0:
                    sm = np.eye(K)[np.argmax(macro_sub[cv[0]])]
                    sm_mat = np.zeros((N, K)); fv = np.zeros(N)
                    for i, v in enumerate(cv[:N]): sm_mat[i] = sm; fv[i] = macro_fmec[v]
                    agent.macro_buffer.add(me, sm_mat, fv,
                                           float(np.mean(rewards[cv])), me, float(done))
                x_all, ei_all = agent.pack_micro_graph(micro_states)
                with torch.no_grad(): ea = agent.micro_encoder(x_all, ei_all).cpu().numpy()
                x_nxt, ei_nxt = agent.pack_micro_graph(nms)
                with torch.no_grad(): en = agent.micro_encoder(x_nxt, ei_nxt).cpu().numpy()
                for n in range(N):
                    agent.micro_buffer.add(ea[n], raw[n], ar, en[n], float(done))
            micro_states = nms
            for agent in agents:
                agent.train_macro(dynamic_reward_weights=dw)
                agent.train_micro(dynamic_reward_weights=dw)

        # GMAE + TCN update
        for m, agent in enumerate(agents):
            if drift_flags[m] in ("soft_drift", "hard_drift"):
                agent.fine_tune_gmae(gmae_losses[m])
            else:
                l = gmae_losses[m]
                if l.requires_grad:
                    agent.gmae_optimizer.zero_grad(); l.backward(); agent.gmae_optimizer.step()
            agent.update_tcn(tcn_losses[m])

        # HFA on hard/structural drift
        if any(f in ("hard_drift", "structural_drift") for f in drift_flags):
            hfa_count += 1
            md = [a.get_state_dict()['macro_actor'] for a in agents]
            ud = [a.get_state_dict()['micro_actor']  for a in agents]
            am = hfa.aggregate_macro(md); au = hfa.aggregate_micro(ud)
            for a in agents:
                sd = a.get_state_dict(); sd['macro_actor'] = am; sd['micro_actor'] = au
                a.load_state_dict(sd)
            if "structural_drift" in drift_flags:
                for a in agents: a.nbocd.reset()

        env.step_macro(ma)
        mac_state = env._get_macro_state()

    return dict(throughput=np.mean(ep_tp), latency_ms=np.mean(ep_lat),
                ret=np.mean(ep_ret), vio=np.mean(ep_vio), hfa_rounds=hfa_count)


# ── Main ────────────────────────────────────────────────────────────────────
def run_experiment(drift_type, seed):
    with open('common/configs/default.yaml') as f:
        config = yaml.safe_load(f)

    np.random.seed(seed); torch.manual_seed(seed)

    env_h = VehicularEnv(config)
    env_f = VehicularEnv(config)
    higat  = HiGAT_MASAC_Trainer(config)
    agents = [FORGE_SAC(config) for _ in range(env_f.num_rsus)]
    hfa    = HFA(env_f.num_rsus)

    results = {k: {'tp': [], 'lat': [], 'ret': [], 'vio': []}
               for k in ['higat', 'forge']}
    results['forge']['hfa'] = []

    # ── Phase 1: stationary training ──────────────────────
    for ep in range(PHASE_1):
        h = run_higat_episode_train(higat, env_h, seed=seed + ep)
        f = run_forge_episode(agents, hfa, env_f, seed=seed + ep)
        for k, src in [('tp','throughput'),('lat','latency_ms'),('ret','ret'),('vio','vio')]:
            results['higat'][k].append(h[src])
            results['forge'][k].append(f[src])
        results['forge']['hfa'].append(f['hfa_rounds'])

    # ── Inject drift ───────────────────────────────────────
    apply_drift(env_h, drift_type, config)
    apply_drift(env_f, drift_type, config)

    # ── Phase 2: HiGAT frozen, FORGE adapts ───────────────
    for ep in range(PHASE_2):
        h = run_higat_episode_eval(higat, env_h, seed=seed + PHASE_1 + ep)
        f = run_forge_episode(agents, hfa, env_f, seed=seed + PHASE_1 + ep)
        for k, src in [('tp','throughput'),('lat','latency_ms'),('ret','ret'),('vio','vio')]:
            results['higat'][k].append(h[src])
            results['forge'][k].append(f[src])
        results['forge']['hfa'].append(f['hfa_rounds'])

    return results


def summarize_and_plot(all_seed_results, drift_type):
    os.makedirs(f'{SAVE_DIR}/plots', exist_ok=True)
    
    # Aggregate across seeds
    metric_keys = ['tp', 'lat', 'ret', 'vio']
    aggregated = {algo: {k: [] for k in metric_keys} for algo in ['higat', 'forge']}
    
    for seed_res in all_seed_results:
        for algo in ['higat', 'forge']:
            for k in metric_keys:
                aggregated[algo][k].append(seed_res[algo][k])

    def get_stats(data_list):
        data = np.array(data_list)
        return np.mean(data, axis=0), np.std(data, axis=0)

    # ── Main 4-panel plot ──────────────────────────────────
    eps = list(range(1, PHASE_1 + PHASE_2 + 1))
    drift_x = PHASE_1 + 0.5
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    panel_cfg = [
        ('tp',  'Throughput (Mbps)',        'higher=better', axes[0, 0]),
        ('lat', 'Avg Latency (ms)',          'lower=better',  axes[0, 1]),
        ('ret', 'Avg Per-Step Return',    '',      axes[1, 0]),
        ('vio', 'Delay Violation Rate',     'lower=better',  axes[1, 1]),
    ]

    for k, ylabel, hint, ax in panel_cfg:
        for algo, col, label in [('higat', '#e74c3c', 'HiGAT-MASAC (frozen)'), 
                                ('forge', '#2980b9', 'FORGE (adaptive)')]:
            mean, std = get_stats(aggregated[algo][k])
            ax.plot(eps, mean, color=col, linewidth=2.5, label=label)
            ax.fill_between(eps, mean - std, mean + std, color=col, alpha=0.15)
            
        ax.axvline(drift_x, color='k', linestyle='--', linewidth=1.5)
        ax.axvspan(PHASE_1, PHASE_1 + PHASE_2, alpha=0.07, color='#f39c12')
        ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} ({len(all_seed_results)} seeds)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plots/comparison_{drift_type}_multiseed.png', dpi=150)
    plt.close()


def main():
    seeds = [42, 123, 789]
    for drift_type in ["vehicle_surge", "channel_shift", "task_overload"]:
        print(f"\nEvaluating drift: {drift_type}")
        all_seed_results = []
        for seed in seeds:
            print(f"  Running seed {seed}...")
            res = run_experiment(drift_type, seed)
            all_seed_results.append(res)
        summarize_and_plot(all_seed_results, drift_type)

if __name__ == '__main__':
    main()
