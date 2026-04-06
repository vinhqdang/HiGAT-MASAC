"""
FORGE Soft Actor-Critic (FORGE_SAC)

Full implementation of the FORGE MARL agent with:
  - HiGAT Encoder (reused from HiGAT-MASAC)
  - GMAE-CPD drift detection
  - ERPG evolutive reward weighting
  - Full SAC Macro + Micro update loops (adapted with dynamic ERPG rewards)
  - Drift-prioritized ring buffer for GMAE experience replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Reuse core networks from HiGAT-MASAC
from algorithms.higat_masac.models.gat_encoder import GATEncoder
from algorithms.higat_masac.models.sac_micro import MicroActor, MicroCritic
from algorithms.higat_masac.models.sac_macro import MacroActor, MacroCritic
from algorithms.higat_masac.models.higat_masac import ReplayBuffer, soft_update

from .gmae_cpd import GMAE_CPD, NBOCD
from .erpg import ERPG


# ---------------------------------------------------------------------------
# Drift-Prioritized Ring Buffer (for GMAE replay)
# ---------------------------------------------------------------------------
class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, graph_state, xi):
        data = (graph_state, xi)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        
        n_samples = min(batch_size, len(self.buffer))
        xis = np.array([item[1] for item in self.buffer], dtype=np.float64)
        xis = np.clip(xis, 0, None)
        
        total = xis.sum()
        non_zero_count = np.count_nonzero(xis)
        
        if total <= 0 or non_zero_count < n_samples:
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs = xis / total

        indices = np.random.choice(
            len(self.buffer), n_samples,
            p=probs, replace=False
        )
        return [self.buffer[i] for i in indices]


# ---------------------------------------------------------------------------
# FORGE_SAC Agent
# ---------------------------------------------------------------------------
class FORGE_SAC:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = config['rl']['gamma']
        self.tau   = config['rl']['tau']
        actor_lr   = config['rl']['actor_lr']
        critic_lr  = config['rl']['critic_lr']
        self.batch_size = config['rl']['batch_size']
        self.lambda_mem = 0.5  # Weight for GMAE memory replay loss

        # Ablation flags
        self.use_cpd  = config.get('ablation', {}).get('use_cpd', True)
        self.use_erpg = config.get('ablation', {}).get('use_erpg', True)

        num_vehicles_max = config['env']['num_vehicles']
        num_subbands     = config['env']['num_subbands']
        max_f_mec        = config['env']['rsu_mec_capacity_ghz']
        gat_embed_dim    = config['rl']['gat_embed_dim']

        self.num_vehicles = num_vehicles_max
        self.num_subbands = num_subbands

        macro_in_channels = 5  # v2i_gain, data_size, cpu_intensity, vx, vy
        micro_in_channels = 4  # v2i_gain, data_size, env_subband, env_fmec

        # --- HiGAT Encoders ---
        self.macro_encoder = GATEncoder(macro_in_channels, config).to(self.device)
        self.micro_encoder  = GATEncoder(micro_in_channels,  config).to(self.device)

        # --- Macro (RSU-level) Actor / Twin Critics ---
        self.macro_actor          = MacroActor(gat_embed_dim, num_vehicles_max, num_subbands, max_f_mec).to(self.device)
        self.macro_critic         = MacroCritic(gat_embed_dim, num_vehicles_max, num_subbands).to(self.device)
        self.macro_critic_target  = MacroCritic(gat_embed_dim, num_vehicles_max, num_subbands).to(self.device)
        self.macro_critic_target.load_state_dict(self.macro_critic.state_dict())

        # --- Micro (Vehicle-level) Actor / Twin Critics ---
        self.micro_actor          = MicroActor(gat_embed_dim, 2, max_action=1.0).to(self.device)
        self.micro_critic         = MicroCritic(gat_embed_dim, 2).to(self.device)
        self.micro_critic_target  = MicroCritic(gat_embed_dim, 2).to(self.device)
        self.micro_critic_target.load_state_dict(self.micro_critic.state_dict())

        # --- FORGE Components ---
        self.gmae_cpd      = GMAE_CPD(macro_in_channels, config).to(self.device)
        self.nbocd         = NBOCD(embed_dim=gat_embed_dim).to(self.device)
        self.erpg          = ERPG().to(self.device)
        self.drift_buffer  = RingBuffer(capacity=1000)

        # --- Optimizers ---
        self.macro_actor_optimizer  = optim.Adam(self.macro_actor.parameters(),  lr=actor_lr)
        self.macro_critic_optimizer = optim.Adam(self.macro_critic.parameters(), lr=critic_lr)
        self.micro_actor_optimizer  = optim.Adam(self.micro_actor.parameters(),  lr=actor_lr)
        self.micro_critic_optimizer = optim.Adam(self.micro_critic.parameters(), lr=critic_lr)

        all_encoder_params = list(self.macro_encoder.parameters()) + list(self.micro_encoder.parameters())
        self.encoder_optimizer = optim.Adam(all_encoder_params, lr=actor_lr)

        self.gmae_optimizer = optim.Adam(self.gmae_cpd.parameters(), lr=1e-4)
        self.tcn_optimizer  = optim.Adam(self.nbocd.tcn.parameters(), lr=1e-4)
        self.erpg_optimizer = optim.Adam(self.erpg.meta_controller.parameters(), lr=1e-3)

        # --- Automatic Entropy Tuning ---
        self.target_entropy_macro = -float(num_vehicles_max * num_subbands + num_vehicles_max)
        self.log_alpha_macro      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_macro_optim    = optim.Adam([self.log_alpha_macro], lr=actor_lr)

        self.target_entropy_micro = -2.0
        self.log_alpha_micro      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_micro_optim    = optim.Adam([self.log_alpha_micro], lr=actor_lr)

        # --- Replay Buffers ---
        self.macro_buffer = ReplayBuffer(
            config['rl']['buffer_size'], gat_embed_dim,
            is_macro=True, num_vehicles=num_vehicles_max, num_subbands=num_subbands
        )
        self.micro_buffer = ReplayBuffer(
            config['rl']['buffer_size'], gat_embed_dim, action_dim=2
        )

    # -----------------------------------------------------------------------
    # Graph packing utilities
    # -----------------------------------------------------------------------
    def pack_macro_graph(self, mac_state):
        """Convert list of RSU state dicts to (x, edge_index, batch) tensors."""
        all_x, batch = [], []
        for d in mac_state:
            feats = d['node_features']
            if len(feats) == 0:
                continue
            all_x.append(feats)
            batch.extend([d['rsu_id']] * len(feats))

        if len(all_x) == 0:
            x = torch.zeros((1, 5), dtype=torch.float32, device=self.device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            batch_t = torch.zeros(1, dtype=torch.long, device=self.device)
            return x, edge_index, batch_t

        x = torch.tensor(np.vstack(all_x), dtype=torch.float32, device=self.device)

        edge_index, offset = [], 0
        for d in mac_state:
            N = len(d['node_features'])
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edge_index.append([i + offset, j + offset])
            offset += N

        if edge_index:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        else:
            edge_index_t = torch.empty((2, 0), dtype=torch.long, device=self.device)

        batch_t = torch.tensor(batch, dtype=torch.long, device=self.device)
        return x, edge_index_t, batch_t

    def pack_micro_graph(self, micro_states):
        """Convert list of vehicle micro state dicts to (x, edge_index) tensors."""
        all_x   = [s['local_features'] for s in micro_states]
        x       = torch.tensor(np.array(all_x), dtype=torch.float32, device=self.device)
        N       = x.shape[0]
        # Fully-connected ego-graph for vehicle neighborhood
        edge_index = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index.append([i, j])
        if edge_index:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        else:
            edge_index_t = torch.empty((2, 0), dtype=torch.long, device=self.device)
        return x, edge_index_t

    # -----------------------------------------------------------------------
    # Action Selection
    # -----------------------------------------------------------------------
    def select_macro_action(self, mac_state):
        """Returns (subband_np, fmec_np) arrays from macro actor."""
        x, edge_index, batch = self.pack_macro_graph(mac_state)
        with torch.no_grad():
            state_emb = self.macro_encoder(x, edge_index, batch=batch)
            if state_emb.shape[0] > 1:
                state_emb = state_emb.mean(0, keepdim=True)
            subband, fmec_norm, _, _ = self.macro_actor.sample(state_emb)
        subband_np = subband.cpu().numpy().squeeze(0)
        # Scale sigmoid [0,1] fmec output to actual Hz
        fmec_np = fmec_norm.cpu().numpy().squeeze(0) * self.config['env']['rsu_mec_capacity_ghz'] * 1e9
        fmec_np = np.clip(fmec_np, 1e8, self.config['env']['rsu_mec_capacity_ghz'] * 1e9)
        return subband_np, fmec_np

    def select_micro_action(self, micro_states, evaluate=False):
        """Returns action_np array for all vehicles from micro actor."""
        x, edge_index = self.pack_micro_graph(micro_states)
        with torch.no_grad():
            state_emb = self.micro_encoder(x, edge_index)   # [N, embed_dim]
            action, _, mean_action = self.micro_actor.sample(state_emb)
        if evaluate:
            return mean_action.cpu().numpy()
        return action.cpu().numpy()

    # -----------------------------------------------------------------------
    # Drift Detection
    # -----------------------------------------------------------------------
    def detect_drift(self, mac_state):
        """Run GMAE-CPD + NBOCD on current macro graph. Returns (status, gmae_loss, tcn_loss)."""
        x, edge_index, _ = self.pack_macro_graph(mac_state)

        # Forward GMAE
        loss_gmae, xi, z_pool = self.gmae_cpd(x, edge_index)

        if not self.use_cpd:
            return "no_drift", loss_gmae, torch.tensor(0.0, device=self.device)

        # Update NBOCD with detached signals (separate computation graph)
        cp_prob, r_hat, loss_tcn = self.nbocd.update(xi.detach(), z_pool.detach())

        # Store in drift ring buffer (detached graph state)
        self.drift_buffer.add((x.detach().cpu(), edge_index.detach().cpu()), xi.item())

        # Threshold classification (from forge.md)
        eps_0     = 0.2
        eps_1     = 0.6
        tau_short = 5
        xi_struct = 1.0

        status = "no_drift"
        if cp_prob >= eps_1 and xi.item() > xi_struct:
            status = "structural_drift"
        elif cp_prob >= eps_1 and r_hat < tau_short:
            status = "hard_drift"
        elif cp_prob >= eps_0:
            status = "soft_drift"

        return status, loss_gmae, loss_tcn

    # -----------------------------------------------------------------------
    # GMAE Fine-Tuning (on soft / hard drift)
    # -----------------------------------------------------------------------
    def fine_tune_gmae(self, current_loss_gmae):
        """
        L_total = L_GMAE^current + λ_mem * L_GMAE^replay
        Performs a single gradient step on the GMAE.
        """
        self.gmae_optimizer.zero_grad()

        total_loss = current_loss_gmae  # current graph loss (already has grad_fn)

        # Replay branch
        samples = self.drift_buffer.sample(max(1, self.batch_size // 4))
        if samples:
            replay_loss = torch.tensor(0.0, device=self.device)
            for item in samples:
                x_r, ei_r = item[0]
                l_r, _, _ = self.gmae_cpd(x_r.to(self.device), ei_r.to(self.device))
                replay_loss = replay_loss + l_r / len(samples)
            total_loss = total_loss + self.lambda_mem * replay_loss

        if total_loss.requires_grad:
            total_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.gmae_cpd.parameters(), 1.0)
            self.gmae_optimizer.step()

    def update_tcn(self, loss_tcn):
        if torch.is_tensor(loss_tcn) and loss_tcn.requires_grad:
            self.tcn_optimizer.zero_grad()
            loss_tcn.backward()
            torch.nn.utils.clip_grad_norm_(self.nbocd.tcn.parameters(), 1.0)
            self.tcn_optimizer.step()

    # -----------------------------------------------------------------------
    # SAC Macro Update (ERPG-augmented reward)
    # -----------------------------------------------------------------------
    def train_macro(self, dynamic_reward_weights=None):
        if self.macro_buffer.size < self.batch_size:
            return None

        state, subband, fmec, reward, next_state, done = self.macro_buffer.sample(self.batch_size)
        state      = state.to(self.device)
        subband    = subband.to(self.device)
        fmec       = fmec.to(self.device)
        reward     = reward.to(self.device)
        next_state = next_state.to(self.device)
        done       = done.to(self.device)

        # Optionally re-scale reward via ERPG weights (α already applied before storage,
        # but we can apply an online correction factor here)
        if dynamic_reward_weights is not None:
            alpha_w = dynamic_reward_weights[0].detach()
            reward = reward * alpha_w  # soft scaling to bias toward current metric priority

        alpha_macro = self.log_alpha_macro.exp().detach()

        with torch.no_grad():
            next_sub, next_fmec, next_log_prob, _ = self.macro_actor.sample(next_state)
            tQ1, tQ2 = self.macro_critic_target(next_state, next_sub, next_fmec)
            target_Q  = torch.min(tQ1, tQ2) - alpha_macro * next_log_prob
            target_Q  = reward + (1.0 - done) * self.gamma * target_Q

        # Critic loss
        cQ1, cQ2  = self.macro_critic(state, subband, fmec)
        critic_loss = F.mse_loss(cQ1, target_Q) + F.mse_loss(cQ2, target_Q)
        self.macro_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.macro_critic.parameters(), 1.0)
        self.macro_critic_optimizer.step()

        # Actor loss
        new_sub, new_fmec, log_prob, _ = self.macro_actor.sample(state)
        aQ1, aQ2  = self.macro_critic(state, new_sub, new_fmec)
        actor_loss = (alpha_macro * log_prob - torch.min(aQ1, aQ2)).mean()
        self.macro_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.macro_actor.parameters(), 1.0)
        self.macro_actor_optimizer.step()

        # Alpha (entropy temperature) update
        alpha_loss = (self.log_alpha_macro * (-log_prob.detach() - self.target_entropy_macro)).mean()
        self.alpha_macro_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_macro_optim.step()

        soft_update(self.macro_critic_target, self.macro_critic, self.tau)

        return critic_loss.item()

    # -----------------------------------------------------------------------
    # SAC Micro Update (ERPG-augmented reward)
    # -----------------------------------------------------------------------
    def train_micro(self, dynamic_reward_weights=None):
        if self.micro_buffer.size < self.batch_size:
            return None

        state, action, reward, next_state, done = self.micro_buffer.sample(self.batch_size)
        state      = state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device)
        next_state = next_state.to(self.device)
        done       = done.to(self.device)

        if dynamic_reward_weights is not None:
            gamma_w = dynamic_reward_weights[2].detach()
            reward  = reward * gamma_w

        alpha_micro = self.log_alpha_micro.exp().detach()

        with torch.no_grad():
            next_action, next_log_prob, _ = self.micro_actor.sample(next_state)
            tQ1, tQ2   = self.micro_critic_target(next_state, next_action)
            target_Q   = torch.min(tQ1, tQ2) - alpha_micro * next_log_prob
            target_Q   = reward + (1.0 - done) * self.gamma * target_Q

        cQ1, cQ2     = self.micro_critic(state, action)
        critic_loss  = F.mse_loss(cQ1, target_Q) + F.mse_loss(cQ2, target_Q)
        self.micro_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.micro_critic.parameters(), 1.0)
        self.micro_critic_optimizer.step()

        new_action, log_prob, _ = self.micro_actor.sample(state)
        aQ1, aQ2    = self.micro_critic(state, new_action)
        actor_loss  = (alpha_micro * log_prob - torch.min(aQ1, aQ2)).mean()
        self.micro_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.micro_actor.parameters(), 1.0)
        self.micro_actor_optimizer.step()

        alpha_loss = (self.log_alpha_micro * (-log_prob.detach() - self.target_entropy_micro)).mean()
        self.alpha_micro_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_micro_optim.step()

        soft_update(self.micro_critic_target, self.micro_critic, self.tau)

        return critic_loss.item()

    # -----------------------------------------------------------------------
    # Federated state dict access
    # -----------------------------------------------------------------------
    def get_state_dict(self):
        return {
            'macro_actor':  self.macro_actor.state_dict(),
            'micro_actor':  self.micro_actor.state_dict(),
            'macro_critic': self.macro_critic.state_dict(),
            'micro_critic': self.micro_critic.state_dict(),
        }

    def load_state_dict(self, sd):
        self.macro_actor.load_state_dict(sd['macro_actor'])
        self.micro_actor.load_state_dict(sd['micro_actor'])
        self.macro_critic.load_state_dict(sd['macro_critic'])
        self.micro_critic.load_state_dict(sd['micro_critic'])
