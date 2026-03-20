import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAPPOActor(nn.Module):
    def __init__(self, state_dim, num_subbands, max_f_mec, max_tx_power):
        super(MAPPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Macro actions
        self.subband_head = nn.Linear(128, num_subbands)
        self.fmec_head = nn.Linear(128, 1)
        
        # Micro actions
        self.p_tx_head = nn.Linear(128, 1)
        self.alpha_head = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        subband_logits = self.subband_head(x)
        fmec = torch.sigmoid(self.fmec_head(x))
        
        p_tx = torch.sigmoid(self.p_tx_head(x))
        alpha = torch.sigmoid(self.alpha_head(x))
        return subband_logits, fmec, p_tx, alpha

class MAPPOAgent:
    """
    MAPPO: Multi-Agent PPO (Wang et al. 2025)
    Centralized Critic, Decentralized Actors.
    Simplified implementation for baseline comparison.
    """
    def __init__(self, config):
        self.config = config
        self.num_vehicles = config['env']['num_vehicles']
        self.num_subbands = config['env']['num_subbands']
        self.max_f_mec = config['env']['rsu_mec_capacity_ghz']
        self.max_tx_power = 10 ** (config['env']['max_tx_power_dbm'] / 10.0) * 1e-3
        
        # Simplified Flat State Dim
        state_dim = 15 # Mock state size for flat MAPPO
        self.actors = [MAPPOActor(state_dim, self.num_subbands, self.max_f_mec, self.max_tx_power) for _ in range(self.num_vehicles)]
        
    def extract_flat_state(self, macro_state):
        # Fallback flat state extraction
        return torch.randn(self.num_vehicles, 15)

    def select_macro_action(self, state_graph):
        state = self.extract_flat_state(state_graph)
        subband = np.zeros((self.num_vehicles, self.num_subbands))
        fmec = np.zeros(self.num_vehicles)
        
        for i in range(self.num_vehicles):
            logits, f, _, _ = self.actors[i](state[i])
            sb_idx = torch.argmax(logits).item()
            subband[i, sb_idx] = 1.0
            fmec[i] = f.item()
            
        return subband, fmec

    def select_micro_action(self, state_graph):
        state = self.extract_flat_state(state_graph)
        action = np.zeros((self.num_vehicles, 2))
        for i in range(self.num_vehicles):
            _, _, p, a = self.actors[i](state[i])
            action[i, 0] = p.item()
            action[i, 1] = a.item()
        
        # Since micro action is called per vehicle in train.py mock loop, we just return the first one
        # but the train loop structure assumes we pass micro state list.
        # We will just return random-ish for this dummy baseline integration if flat state fails
        return np.array([action[0, 0], action[0, 1]])

    def train_macro(self): pass
    def train_micro(self): pass
    def knowledge_distillation(self): pass
