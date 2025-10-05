import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network with dimensionality expansion"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Typical expansion: d_ff = 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Expand: [batch, seq_len, d_ff]
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation (used in BERT, GPT)
        x = self.dropout(x)
        # Project back: [batch, seq_len, d_model]
        x = self.linear2(x)
        x = self.dropout(x)
        return x