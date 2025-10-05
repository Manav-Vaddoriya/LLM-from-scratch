import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer Normalization (implemented from scratch for understanding)"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Compute mean and variance along the feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta
