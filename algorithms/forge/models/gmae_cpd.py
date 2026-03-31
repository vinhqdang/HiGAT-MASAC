import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.higat_masac.models.gat_encoder import GATEncoder

class OutputHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.PReLU(),
            nn.Linear(in_dim * 2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class GMAE_CPD(nn.Module):
    def __init__(self, in_channels, config):
        """
        Graph Masked Autoencoder with Attention-Weighted Cosine Error
        for drift detection.
        """
        super(GMAE_CPD, self).__init__()
        self.config = config
        self.mask_ratio = 0.4
        
        # Learnable mask token to replace masked node features
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.xavier_normal_(self.mask_token)
        
        # GAT Encoder to learn from masked graphs
        self.encoder = GATEncoder(in_channels, config)
        embed_dim = config['rl']['gat_embed_dim']
        
        # Decoder reconstructs the original features
        self.decoder = OutputHead(embed_dim, in_channels)
        
        # Attention pooling for scalar drift score
        self.wa = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        num_nodes = x.shape[0]
        
        # 1. Masking
        num_masked = int(self.mask_ratio * num_nodes)
        if num_masked > 0:
            perm = torch.randperm(num_nodes, device=x.device)
            mask_idx = perm[:num_masked]
            
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
            mask[mask_idx] = True
            x_masked = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        else:
            x_masked = x
            mask_idx = torch.arange(0, device=x.device) # empty

        # 2. Encode
        z = self.encoder(x_masked, edge_index, edge_attr=edge_attr)
        
        # 3. Decode
        x_hat = self.decoder(z)
        
        # 4. Scaled Cosine Error for masked nodes
        # Add epsilon to prevent division by zero
        x_norm = F.normalize(x + 1e-8, p=2, dim=-1)
        x_hat_norm = F.normalize(x_hat + 1e-8, p=2, dim=-1)
        # cosine sim = dot product of l2 normalized vectors
        cos_sim = (x_norm * x_hat_norm).sum(dim=-1)
        l_i = 1.0 - cos_sim # Reconstruction error [num_nodes]
        
        # For training the GMAE, we sum/mean the error over masked nodes
        if num_masked > 0:
            loss_gmae = l_i[mask_idx].mean()
        else:
            loss_gmae = torch.tensor(0.0, device=x.device, requires_grad=True)

        # 5. Drift Score (Attention-weighted pooling of errors across ALL nodes)
        a_i_logits = self.wa(z).squeeze(-1) # [num_nodes]
        a_i = torch.softmax(a_i_logits, dim=0) # [num_nodes]
        
        xi = torch.sum(a_i * l_i) # Scalar drift score
        
        # Global pooled representation (for hazard function)
        z_pool = torch.sum(a_i.unsqueeze(-1) * z, dim=0)
        
        return loss_gmae, xi, z_pool


class TCNPredictor(nn.Module):
    def __init__(self, seq_len=10, hidden_dim=32):
        super().__init__()
        # Simple causal 1D Conv as TCN
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2)
        # Outputs mean and log_var of next xi
        self.fc = nn.Linear(hidden_dim * seq_len, 2)
        self.seq_len = seq_len
        
    def forward(self, xi_history):
        # xi_history: [batch, seq_len]
        x = xi_history.unsqueeze(1) # [batch, 1, seq_len]
        
        # Causal convolution (trimming output)
        x = F.relu(self.conv1(x))[:, :, :-2]
        x = F.relu(self.conv2(x))[:, :, :-2]
        
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        mu_xi = out[:, 0]
        logvar_xi = out[:, 1]
        
        return mu_xi, logvar_xi

class NBOCD(nn.Module):
    def __init__(self, embed_dim, seq_len=10, max_run_length=1000):
        super(NBOCD, self).__init__()
        self.tcn = TCNPredictor(seq_len=seq_len)
        
        # Hazard function mapping [z_pool, run_length] -> hazard prob
        self.hazard_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.max_run_length = max_run_length
        self.seq_len = seq_len
        
        # Internal state
        self.reset()
        
    def reset(self):
        self.r_posterior = torch.zeros(self.max_run_length)
        self.r_posterior[0] = 1.0 # P(r_0 = 0) = 1
        self.history = []
        
    def update(self, xi, z_pool):
        # xi: scalar drift score
        # z_pool: [embed_dim]
        
        device = z_pool.device
        self.r_posterior = self.r_posterior.to(device)
        
        self.history.append(xi.item())
        if len(self.history) > self.seq_len:
            self.history.pop(0)
            
        # We need seq_len history to predict using TCN
        if len(self.history) < self.seq_len:
            # Not enough history, append deterministic
            cp_prob = 0.0
            r_hat = len(self.history)
            
            # Shift posterior safely
            prior = self.r_posterior.clone()
            self.r_posterior = torch.zeros_like(prior)
            self.r_posterior[1:] = prior[:-1]
            return cp_prob, r_hat, torch.tensor(0.0, device=device, requires_grad=True)
            
        # 1. Predictive Likelihood P(xi_t | r_{t-1}, History)
        history_tensor = torch.tensor(self.history, dtype=torch.float32, device=device).unsqueeze(0)
        mu, logvar = self.tcn(history_tensor)
        var = torch.exp(logvar)
        
        # Gaussian log likelihood
        nll = 0.5 * ((xi - mu)**2 / (var + 1e-6) + logvar + torch.log(torch.tensor(2 * torch.pi)))
        likelihood = torch.exp(-nll).squeeze() # scalar
        
        loss_tcn = nll.mean() # For training the TCN
        
        # 2. Hazard Function
        # For simplicity in PyTorch, we compute hazard based on the most likely previous run_length
        r_prev = torch.argmax(self.r_posterior).item()
        hazard_input = torch.cat([z_pool, torch.tensor([float(r_prev)], device=device)])
        h_t = self.hazard_net(hazard_input) # scalar probability
        
        # 3. Posterior Update
        prior = self.r_posterior.clone()
        new_posterior = torch.zeros_like(prior)
        
        # P(r_t = 0) = sum(prior * h_t * likelihood) -> If change point
        new_posterior[0] = torch.sum(prior * h_t * likelihood)
        
        # P(r_t > 0) = prior * (1 - h_t) * likelihood -> If no change point
        new_posterior[1:] = prior[:-1] * (1 - h_t) * likelihood
        
        # Normalize
        self.r_posterior = new_posterior / (torch.sum(new_posterior) + 1e-12)
        
        cp_prob = self.r_posterior[0].item()
        r_hat = torch.argmax(self.r_posterior).item()
        
        return cp_prob, r_hat, loss_tcn
