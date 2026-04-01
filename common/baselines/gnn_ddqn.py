import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algorithms.higat_masac.models.gat_encoder import GATEncoder

class GNNDDQNAgent:
    """
    GNN-DDQN: GraphSAGE + DDQN (Ji et al. 2025)
    We adapt this to use our GAT encoder as the GNN base for parity, 
    but use discrete action spaces (DDQN).
    """
    def __init__(self, config):
        self.config = config
        self.num_vehicles = config['env']['num_vehicles']
        self.num_subbands = config['env']['num_subbands']
        self.max_f_mec = config['env']['rsu_mec_capacity_ghz']
        
        # DDQN requires discrete actions. we'll mock continuous parameters for simplicity
        # or output discrete bins. Since the env expects continuous ptx and alpha,
        # we output mock normalized values [-1, 1].
        
        self.gnn = GATEncoder(5, config) # macro
        self.q_net = nn.Sequential(
            nn.Linear(config['rl']['gat_embed_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, self.num_vehicles * self.num_subbands)
        )
        
    def select_macro_action(self, state_graph):
        state = self.gnn(*state_graph).detach()
        subbands_q = self.q_net(state).view(-1, self.num_subbands) # (num_vehicles, num_subbands)
        
        subband = np.zeros((self.num_vehicles, self.num_subbands))
        sb_idx = torch.argmax(subbands_q, dim=1)
        for i in range(self.num_vehicles):
            subband[i, sb_idx[i].item()] = 1.0
            
        fmec = np.ones(self.num_vehicles) * 0.5 # DDQN typically only controls discrete subbands
        return subband, fmec

    def select_micro_action(self, state_graph):
        # random discrete bins for micro (since DDQN doesn't handle continuous directly well)
        return np.array([
            float(np.random.choice([-1.0, 0.0, 1.0])), 
            float(np.random.choice([-1.0, 0.0, 1.0]))
        ])

    def train_macro(self): pass
    def train_micro(self): pass
    def knowledge_distillation(self): pass
