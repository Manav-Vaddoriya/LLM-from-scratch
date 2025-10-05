import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1.1 POSITIONAL EMBEDDINGS
# ============================================================================

class LearnedPositionalEmbedding(nn.Module):
    """Absolute learned positional embeddings"""
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (original Transformer paper)"""
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        # Create matrix of [seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]