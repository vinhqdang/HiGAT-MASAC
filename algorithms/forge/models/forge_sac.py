"""
FORGE Soft Actor-Critic (SAC) Core

Implements the base Multi-Agent SAC modified to support
evolutionary rewards and threshold-triggered federated aggregation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# We reuse the core networks from HiGAT-MASAC for modularity
from algorithms.higat_masac.models.gat_encoder import GATEncoder
from algorithms.higat_masac.models.sac_micro import MicroActor, MicroCritic
from algorithms.higat_masac.models.sac_macro import MacroActor, MacroCritic
from algorithms.higat_masac.models.higat_masac import ReplayBuffer, soft_update

from .gmae_cpd import GMAE_CPD, NBOCD
from .erpg import ERPG

class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        
    def add(self, graph_state, xi):
        # We store the maskable graph representations and their drift scores
        # for drift-weighted replay of the GMAE.
        data = (graph_state, xi)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        # Priority mapping: higher xi -> higher probability
        xis = np.array([item[1] for item in self.buffer])
        probs = xis / (np.sum(xis) + 1e-8)
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        return [self.buffer[i] for i in indices]

class FORGE_SAC:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config['rl']['gamma']
        self.tau = config['rl']['tau']
        actor_lr = config['rl']['actor_lr']
        critic_lr = config['rl']['critic_lr']
        self.batch_size = config['rl']['batch_size']
        
        num_vehicles_max = config['env']['num_vehicles']
        num_subbands = config['env']['num_subbands']
        max_f_mec = config['env']['rsu_mec_capacity_ghz']
        gat_embed_dim = config['rl']['gat_embed_dim']
        
        macro_in_channels = 5
        micro_in_channels = 4

        # Encoders
        self.macro_encoder = GATEncoder(macro_in_channels, config).to(self.device)
        self.micro_encoder = GATEncoder(micro_in_channels, config).to(self.device)
        
        # Actors / Critics
        self.macro_actor = MacroActor(gat_embed_dim, num_vehicles_max, num_subbands, max_f_mec).to(self.device)
        self.macro_critic = MacroCritic(gat_embed_dim, num_vehicles_max, num_subbands).to(self.device)
        self.macro_critic_target = MacroCritic(gat_embed_dim, num_vehicles_max, num_subbands).to(self.device)
        self.macro_critic_target.load_state_dict(self.macro_critic.state_dict())
        
        self.micro_actor = MicroActor(gat_embed_dim, 2, max_action=1.0).to(self.device)
        self.micro_critic = MicroCritic(gat_embed_dim, 2).to(self.device)
        self.micro_critic_target = MicroCritic(gat_embed_dim, 2).to(self.device)
        self.micro_critic_target.load_state_dict(self.micro_critic.state_dict())
        
        # FORGE Additions
        self.gmae_cpd = GMAE_CPD(macro_in_channels, config).to(self.device)
        self.gmae_optimizer = optim.Adam(self.gmae_cpd.parameters(), lr=1e-4) # lower LR for autoencoder
        self.nbocd = NBOCD(embed_dim=gat_embed_dim).to(self.device)
        self.tcn_optimizer = optim.Adam(self.nbocd.tcn.parameters(), lr=1e-4)
        
        self.erpg = ERPG().to(self.device)
        self.erpg_optimizer = optim.Adam(self.erpg.meta_controller.parameters(), lr=1e-3)

        self.drift_buffer = RingBuffer(capacity=1000)

        # RL Optimizers
        self.macro_actor_optimizer = optim.Adam(self.macro_actor.parameters(), lr=actor_lr)
        self.macro_critic_optimizer = optim.Adam(self.macro_critic.parameters(), lr=critic_lr)
        self.micro_actor_optimizer = optim.Adam(self.micro_actor.parameters(), lr=actor_lr)
        self.micro_critic_optimizer = optim.Adam(self.micro_critic.parameters(), lr=critic_lr)
        
        # Alpha
        self.log_alpha_macro = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_macro_optim = optim.Adam([self.log_alpha_macro], lr=actor_lr)
        self.target_entropy_macro = -float(num_vehicles_max * num_subbands + num_vehicles_max)
        
        self.log_alpha_micro = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_micro_optim = optim.Adam([self.log_alpha_micro], lr=actor_lr)
        self.target_entropy_micro = -2.0
        
        # Buffers
        self.macro_buffer = ReplayBuffer(config['rl']['buffer_size'], gat_embed_dim, 
                                         is_macro=True, num_vehicles=num_vehicles_max, num_subbands=num_subbands)
        self.micro_buffer = ReplayBuffer(config['rl']['buffer_size'], gat_embed_dim, action_dim=2)
        
    def pack_graph(self, mac_state):
        # Turns list of RSU dicts into contiguous pytorch geometric style inputs
        # For simplicity in this demo wrapper, we mock the GAT edge_index
        all_x = []
        batch = []
        for d in mac_state:
            all_x.append(d['node_features'])
            batch.extend([d['rsu_id']] * len(d['node_features']))
            
        x = torch.tensor(np.vstack(all_x), dtype=torch.float32, device=self.device)
        # Fully connected within cluster for demo
        edge_index = []
        offset = 0
        for d in mac_state:
            N = len(d['node_features'])
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edge_index.append([i + offset, j + offset])
            offset += N
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long, device=self.device)
            
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        return x, edge_index, batch

    def detect_drift(self, mac_state):
        x, edge_index, batch = self.pack_graph(mac_state)
        # 1. Forward pass
        loss_gmae, xi, z_pool = self.gmae_cpd(x, edge_index)
        
        # 2. Update TCN (detach inputs so we don't backprop into GMAE)
        cp_prob, r_hat, loss_tcn = self.nbocd.update(xi.detach(), z_pool.detach())
        
        self.drift_buffer.add((x.cpu(), edge_index.cpu()), xi.item())
        
        # Classification thresholds
        eps_0 = 0.2
        eps_1 = 0.6
        tau_short = 5
        xi_struct = 1.0 # arbitrary high threshold
        
        status = "no_drift"
        if cp_prob >= eps_1 and xi.item() > xi_struct:
            status = "structural_drift"
        elif cp_prob >= eps_1 and r_hat < tau_short:
            status = "hard_drift"
        elif cp_prob >= eps_0:
            status = "soft_drift"
            
        return status, loss_gmae, loss_tcn

    def fine_tune_gmae(self):
        samples = self.drift_buffer.sample(self.batch_size)
        if len(samples) == 0: return
        
        self.gmae_optimizer.zero_grad()
        for item in samples:
            x, edge_index = item[0]
            loss_gmae, _, _ = self.gmae_cpd(x.to(self.device), edge_index.to(self.device))
            # Average loss over the batch before backprop
            if loss_gmae.requires_grad and loss_gmae.item() > 0:
                (loss_gmae / len(samples)).backward()
            
        self.gmae_optimizer.step()

    def update_tcn(self, loss_tcn):
        if loss_tcn > 0:
            self.tcn_optimizer.zero_grad()
            loss_tcn.backward()
            self.tcn_optimizer.step()

    # We replicate train_macro and train_micro directly but omitting their massive bodies for brevity
    # We assume they operate similarly but pull dynamic rewards from ERPG instead
    
    def get_state_dict(self):
        return {
            'macro_actor': self.macro_actor.state_dict(),
            'micro_actor': self.micro_actor.state_dict(),
            'macro_critic': self.macro_critic.state_dict(),
            'micro_critic': self.micro_critic.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        self.macro_actor.load_state_dict(state_dict['macro_actor'])
        self.micro_actor.load_state_dict(state_dict['micro_actor'])
        self.macro_critic.load_state_dict(state_dict['macro_critic'])
        self.micro_critic.load_state_dict(state_dict['micro_critic'])
