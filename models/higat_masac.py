import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .gat_encoder import GATEncoder
from .sac_micro import MicroActor, MicroCritic
from .sac_macro import MacroActor, MacroCritic

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=None, is_macro=False, num_vehicles=0, num_subbands=0):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        
        self.is_macro = is_macro
        if self.is_macro:
            self.subband_action = np.zeros((capacity, num_vehicles, num_subbands), dtype=np.float32)
            self.fmec_action = np.zeros((capacity, num_vehicles), dtype=np.float32)
        else:
            self.action = np.zeros((capacity, action_dim), dtype=np.float32)
            
    def add(self, *args):
        if self.is_macro:
            state, subband, fmec, reward, next_state, done = args
            self.subband_action[self.ptr] = subband
            self.fmec_action[self.ptr] = fmec
        else:
            state, action, reward, next_state, done = args
            self.action[self.ptr] = action
            
        self.state[self.ptr] = state
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        if self.is_macro:
            return (
                torch.FloatTensor(self.state[ind]),
                torch.FloatTensor(self.subband_action[ind]),
                torch.FloatTensor(self.fmec_action[ind]),
                torch.FloatTensor(self.reward[ind]),
                torch.FloatTensor(self.next_state[ind]),
                torch.FloatTensor(self.done[ind])
            )
        else:
            return (
                torch.FloatTensor(self.state[ind]),
                torch.FloatTensor(self.action[ind]),
                torch.FloatTensor(self.reward[ind]),
                torch.FloatTensor(self.next_state[ind]),
                torch.FloatTensor(self.done[ind])
            )

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class HiGAT_MASAC:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config['rl']['gamma']
        self.tau = config['rl']['tau']
        actor_lr = config['rl']['actor_lr']
        critic_lr = config['rl']['critic_lr']
        self.batch_size = config['rl']['batch_size']
        
        # Dimensions
        num_vehicles_max = config['env']['num_vehicles']
        num_subbands = config['env']['num_subbands']
        max_f_mec = config['env']['rsu_mec_capacity_ghz']
        gat_embed_dim = config['rl']['gat_embed_dim']
        
        # Assuming node features have certain dimensions defined by env
        macro_in_channels = 5 # v2i_gain, data, cpu, vx, vy
        micro_in_channels = 4 # v2i, data, subband, fmec

        # Encoders
        self.macro_encoder = GATEncoder(macro_in_channels, config).to(self.device)
        self.micro_encoder = GATEncoder(micro_in_channels, config).to(self.device)
        
        # Macro Agents (RSUs) - Centralized critic for CTDE or independent. Let's do independent per RSU for scalability
        macro_state_dim = gat_embed_dim
        self.macro_actor = MacroActor(macro_state_dim, num_vehicles_max, num_subbands, max_f_mec).to(self.device)
        self.macro_critic = MacroCritic(macro_state_dim, num_vehicles_max, num_subbands).to(self.device)
        self.macro_critic_target = MacroCritic(macro_state_dim, num_vehicles_max, num_subbands).to(self.device)
        self.macro_critic_target.load_state_dict(self.macro_critic.state_dict())
        
        # Micro Agents (Vehicles)
        # Micro state: GAT embedding + 5 x (dist, channel_gain from neighbors) flattened?
        # Actually GAT embeddings incorporate neighbor info, so state_dim = gat_embed_dim
        micro_state_dim = gat_embed_dim
        micro_act_dim = 2 # p_tx, alpha
        self.micro_actor = MicroActor(micro_state_dim, micro_act_dim, max_action=1.0).to(self.device)
        self.micro_critic = MicroCritic(micro_state_dim, micro_act_dim).to(self.device)
        self.micro_critic_target = MicroCritic(micro_state_dim, micro_act_dim).to(self.device)
        self.micro_critic_target.load_state_dict(self.micro_critic.state_dict())
        
        # Optimizers
        self.macro_actor_optimizer = optim.Adam(self.macro_actor.parameters(), lr=actor_lr)
        self.macro_critic_optimizer = optim.Adam(self.macro_critic.parameters(), lr=critic_lr)
        self.micro_actor_optimizer = optim.Adam(self.micro_actor.parameters(), lr=actor_lr)
        self.micro_critic_optimizer = optim.Adam(self.micro_critic.parameters(), lr=critic_lr)
        
        # Automatic Entropy Tuning
        self.target_entropy_macro = -float(num_vehicles_max * num_subbands + num_vehicles_max)
        self.log_alpha_macro = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_macro_optim = optim.Adam([self.log_alpha_macro], lr=actor_lr)
        
        self.target_entropy_micro = -float(micro_act_dim)
        self.log_alpha_micro = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_micro_optim = optim.Adam([self.log_alpha_micro], lr=actor_lr)
        
        # Replay Buffers
        self.macro_buffer = ReplayBuffer(config['rl']['buffer_size'], macro_state_dim, 
                                         is_macro=True, num_vehicles=num_vehicles_max, num_subbands=num_subbands)
        self.micro_buffer = ReplayBuffer(config['rl']['buffer_size'], micro_state_dim, action_dim=micro_act_dim)

    def select_macro_action(self, state_graph):
        state = self.macro_encoder(*state_graph).detach() # pool if needed, handle properly in train.py
        with torch.no_grad():
            subband, fmec, _, _ = self.macro_actor.sample(state)
        return subband.cpu().numpy().squeeze(0), fmec.cpu().numpy().squeeze(0)
        
    def select_micro_action(self, state_graph):
        state = self.micro_encoder(*state_graph).detach()
        with torch.no_grad():
            action, _, _ = self.micro_actor.sample(state)
        return action.cpu().numpy().squeeze(0)
        
    def train_macro(self):
        if self.macro_buffer.size < self.batch_size:
            return
            
        state, subband, fmec, reward, next_state, done = self.macro_buffer.sample(self.batch_size)
        state, subband, fmec, reward, next_state, done = (
            state.to(self.device), subband.to(self.device), fmec.to(self.device),
            reward.to(self.device), next_state.to(self.device), done.to(self.device)
        )
        
        with torch.no_grad():
            next_subband, next_fmec, next_log_prob, _ = self.macro_actor.sample(next_state)
            target_Q1, target_Q2 = self.macro_critic_target(next_state, next_subband, next_fmec)
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha_macro.exp() * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Critic update
        current_Q1, current_Q2 = self.macro_critic(state, subband, fmec)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.macro_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.macro_critic_optimizer.step()

        # Actor update
        new_subband, new_fmec, log_prob, _ = self.macro_actor.sample(state)
        actor_Q1, actor_Q2 = self.macro_critic(state, new_subband, new_fmec)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.log_alpha_macro.exp().detach() * log_prob - actor_Q).mean()

        self.macro_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.macro_actor_optimizer.step()

        # Alpha update
        alpha_loss = (self.log_alpha_macro * (-log_prob - self.target_entropy_macro).detach()).mean()

        self.alpha_macro_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_macro_optim.step()

        soft_update(self.macro_critic_target, self.macro_critic, self.tau)

    def train_micro(self):
        if self.micro_buffer.size < self.batch_size:
            return
            
        state, action, reward, next_state, done = self.micro_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = (
            state.to(self.device), action.to(self.device),
            reward.to(self.device), next_state.to(self.device), done.to(self.device)
        )
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.micro_actor.sample(next_state)
            target_Q1, target_Q2 = self.micro_critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha_micro.exp() * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Critic update
        current_Q1, current_Q2 = self.micro_critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.micro_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.micro_critic_optimizer.step()

        # Actor update
        new_action, log_prob, _ = self.micro_actor.sample(state)
        actor_Q1, actor_Q2 = self.micro_critic(state, new_action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.log_alpha_micro.exp().detach() * log_prob - actor_Q).mean()

        self.micro_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.micro_actor_optimizer.step()

        # Alpha update
        alpha_loss = (self.log_alpha_micro * (-log_prob - self.target_entropy_micro).detach()).mean()

        self.alpha_micro_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_micro_optim.step()

        soft_update(self.micro_critic_target, self.micro_critic, self.tau)
        
    def knowledge_distillation(self):
        # Align GAT parameters across macro and micro tiers
        # Skip the first layer since in_channels differ (5 vs 4)
        for target_conv, source_conv in zip(self.micro_encoder.convs[1:], self.macro_encoder.convs[1:]):
            soft_update(target_conv, source_conv, tau=0.5) 
