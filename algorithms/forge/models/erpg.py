"""
Evolutive Reward and Pseudo-Label Generator (ERPG)

This module implements dynamic reward evolution to adjust weights online.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ERPG(nn.Module):
    def __init__(self, history_len=5):
        super(ERPG, self).__init__()
        
        self.history_len = history_len
        
        # Meta-controller mapping smoothed performance improvement delta R to dynamic weights
        # We look at historical deltas of the 3 metrics: delay, energy, throughput
        self.meta_controller = nn.Sequential(
            nn.Linear(3 * history_len, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
            # Outputs raw scores for alpha (delay), beta (energy), gamma (throughput)
        )
        
        # Keep track of recent performances
        self.metric_history = {
            'delay': [],
            'energy': [],
            'throughput': []
        }
        
        # Default initialization (e.g., balanced)
        self.current_weights = torch.tensor([0.33, 0.33, 0.33])

    def update_history(self, avg_delay, avg_energy, avg_throughput):
        self.metric_history['delay'].append(avg_delay)
        self.metric_history['energy'].append(avg_energy)
        self.metric_history['throughput'].append(avg_throughput)
        
        if len(self.metric_history['delay']) > self.history_len:
            self.metric_history['delay'].pop(0)
            self.metric_history['energy'].pop(0)
            self.metric_history['throughput'].pop(0)

    def evolve_weights(self, device):
        if len(self.metric_history['delay']) < self.history_len:
            # Return uniform weights while warming up
            self.current_weights = torch.tensor([0.33, 0.33, 0.33], device=device)
            return self.current_weights
            
        # Compute deltas (improvements over time)
        # We want decrease in delay/energy, increase in throughput
        d_delays = [- (self.metric_history['delay'][i] - self.metric_history['delay'][i-1]) for i in range(1, self.history_len)]
        d_energies = [- (self.metric_history['energy'][i] - self.metric_history['energy'][i-1]) for i in range(1, self.history_len)]
        d_throughputs = [self.metric_history['throughput'][i] - self.metric_history['throughput'][i-1] for i in range(1, self.history_len)]
        
        # Duplicate last to keep length == history_len
        d_delays.append(d_delays[-1])
        d_energies.append(d_energies[-1])
        d_throughputs.append(d_throughputs[-1])
        
        # Concatenate history
        state_input = torch.tensor(d_delays + d_energies + d_throughputs, dtype=torch.float32, device=device)
        
        # Generate new weights via softmax to ensure sum = 1
        logits = self.meta_controller(state_input)
        self.current_weights = F.softmax(logits, dim=0)
        
        return self.current_weights

    def get_reward(self, delay, energy, throughput):
        """
        Dynamically weights the raw metrics to produce the evolved reward.
        Delay and Energy are penalized, Throughput is rewarded.
        """
        alpha, beta, gamma = self.current_weights[0].item(), self.current_weights[1].item(), self.current_weights[2].item()
        
        r = -alpha * delay - beta * energy + gamma * throughput
        return r

    def generate_pseudo_label(self, value_estimate, empirical_reward):
        """
        Generates pseudo-labels for unlabeled state transitions 
        using the current policy's value estimate.
        
        $\\tilde{y}_{k,m}^t = V_\\psi(s_{k,m}^t) + \\hat{r}_{k,m}^t$
        
        If empirical_reward is missing/masked, we rely largely on V_estimate.
        """
        pseudo_label = value_estimate + empirical_reward
        return pseudo_label
