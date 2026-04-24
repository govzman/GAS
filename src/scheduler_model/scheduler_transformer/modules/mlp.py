import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        num_mlp_layers,
        in_channels,
        hidden_dim,
        num_timesteps
    ): 
        super().__init__()
        
        mlps = []
        mlps.append(nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU()
        ))
        for i in range(num_mlp_layers - 2):
            mlps.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim)
            ))
        mlps.append(nn.Linear(hidden_dim, num_timesteps))
        self.mlps = nn.Sequential(*mlps)

    def forward(self, x):
        return self.mlps(x)
