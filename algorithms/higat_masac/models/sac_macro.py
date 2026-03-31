import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class MacroActor(nn.Module):
    """
    Hybrid Action space:
    - Discrete variable subbands mapping via Gumbel-Softmax
    - Continuous variable computing resources F_mec
    """
    def __init__(self, state_dim, num_vehicles_max, num_subbands, max_f_mec):
        super(MacroActor, self).__init__()
        self.num_vehicles = num_vehicles_max
        self.num_subbands = num_subbands
        self.max_f_mec = max_f_mec
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # For discrete subbands per vehicle
        self.subband_logits = nn.Linear(256, num_vehicles_max * num_subbands)
        
        # For continuous f_mec per vehicle
        self.fmec_mu = nn.Linear(256, num_vehicles_max)
        self.fmec_log_std = nn.Linear(256, num_vehicles_max)

    def forward(self, state):
        x = self.net(state)
        
        subband_logits = self.subband_logits(x).view(-1, self.num_vehicles, self.num_subbands)
        
        fmec_mu = self.fmec_mu(x)
        fmec_log_std = self.fmec_log_std(x)
        fmec_log_std = torch.clamp(fmec_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return subband_logits, fmec_mu, fmec_log_std

    def sample(self, state, tau_gumbel=1.0):
        logits, mu, log_std = self.forward(state)
        
        # Subband Softmax (Gumbel if we need differentiability through discrete)
        # We output probabilities and gumbel sample
        gumbel_sample = F.gumbel_softmax(logits, tau=tau_gumbel, hard=False)
        hard_sample = F.one_hot(torch.argmax(gumbel_sample, dim=-1), num_classes=self.num_subbands).float()
        
        # Straight-through estimator
        subband_action = (hard_sample - gumbel_sample).detach() + gumbel_sample
        
        # Continuous f_mec
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()
        fmec_action = torch.sigmoid(x_t) # normalized [0, 1]
        
        log_prob_f = normal.log_prob(x_t)
        log_prob_f -= torch.log(fmec_action * (1 - fmec_action) + epsilon)
        log_prob_f = log_prob_f.sum(1, keepdim=True)
        
        # Softmax log prob approximation (entropy over classes)
        probs = F.softmax(logits, dim=-1)
        log_prob_s = torch.log(probs + epsilon)
        subband_log_prob = (probs * log_prob_s).sum(-1).sum(-1, keepdim=True)
        
        total_log_prob = log_prob_f + subband_log_prob
        
        # Continuous deterministic mean
        fmec_deterministic = torch.sigmoid(mu)
        
        return subband_action, fmec_action, total_log_prob, fmec_deterministic

class MacroCritic(nn.Module):
    def __init__(self, state_dim, num_vehicles_max, num_subbands):
        super(MacroCritic, self).__init__()
        
        action_dim = num_vehicles_max * num_subbands + num_vehicles_max
        
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, subband_action, fmec_action):
        # Flatten actions
        subband_flat = subband_action.view(subband_action.size(0), -1)
        sa = torch.cat([state, subband_flat, fmec_action], 1)
        
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2
