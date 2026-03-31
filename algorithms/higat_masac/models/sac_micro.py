import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Constants for numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class MicroActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(MicroActor, self).__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick
        action = torch.tanh(x_t)
        
        # Enforcing Action Bounds
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scale action to max_action
        # The actions are p_tx and alpha. We assume they are normalized to [-1, 1] mapped to [0, max]
        # Custom mapping can be done outside, but let's assume we output [-1, 1] generally
        return action, log_prob, torch.tanh(mu)

class MicroCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MicroCritic, self).__init__()

        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2
